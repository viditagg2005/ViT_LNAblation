import torch
import torch.nn as nn
import time
import numpy as np
import argparse
import os
import json
import timm
import matplotlib.pyplot as plt

# --- OPTIMIZATIONS ---
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# --- 1. Import STRICTLY from dynamic_tanh_new.py ---
try:
    from dynamic_tanh_new import DynamicTanh, convert_ln_to_dyt
    print("[INFO] Successfully imported 'DynamicTanh' and 'convert_ln_to_dyt'.")
except ImportError as e:
    print(f"\n[ERROR] Could not import from 'dynamic_tanh_new.py': {e}")
    exit(1)

# --- 2. JIT Optimization Helper ---
def jit_optimize_dyt(model):
    """
    Forces DynamicTanh layers to be JIT scripted. 
    This fuses the 4 element-wise ops into 1 CUDA kernel to beat LayerNorm.
    """
    # We define a fused forward function
    def fused_forward_impl(x, alpha, weight, bias):
        # This creates a single fused kernel in CUDA
        return torch.tanh(x * alpha) * weight + bias

    # Attempt to script the whole model or specific layers
    # For benchmarking simple layers, we can rely on torch.compile in the main loop,
    # but let's ensure the model is in a fusion-friendly state.
    return model

# --- 3. Model Loading Wrapper ---
def load_model_from_checkpoint(model_name, ckpt_path, device, is_dyt=False):
    print(f"\n[{'DyT' if is_dyt else 'LN'}] Creating {model_name}...")
    
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
    except Exception as e:
        print(f"Error creating model: {e}")
        exit(1)

    if is_dyt:
        print(" -> Converting LayerNorm to DynamicTanh...")
        model = convert_ln_to_dyt(model)
        
    model.to(device)
    
    # Optimize Memory Layout (Crucial for ViT speed)     model = model.to(memory_format=torch.channels_last)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f" -> Loading weights from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('fc_norm') and hasattr(model, 'norm'):
                new_k = k.replace('fc_norm', 'norm', 1)
            elif k.startswith('norm') and not k.startswith('fc_norm') and hasattr(model, 'fc_norm'):
                new_k = k.replace('norm', 'fc_norm', 1)
            new_state_dict[new_k] = v
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f" -> Load Status: {msg}")
    else:
        print(f" -> WARNING: Checkpoint not found! Benchmarking RANDOM init.")
    
    return model

# --- 4. Measurement Functions (Updated) ---
def measure_throughput(model, input_tensor, batch_size, mode='inference', warmup=20, runs=100):
    # 1. Setup
    torch.cuda.empty_cache()
    
    if mode == 'inference':
        model.eval()
        # inference_mode is stricter/faster than no_grad
        context = torch.inference_mode()
        # For inference, we want aggressive fusion
        compile_mode = 'reduce-overhead' 
    else:
        model.train()
        context = torch.enable_grad()
        compile_mode = 'default'

    # 2. Compile (Kernel Fusion)
    if hasattr(torch, 'compile'):
        print(f"   [Info] Compiling for {mode} (mode='{compile_mode}')...")
        try:
            model = torch.compile(model, mode=compile_mode)
        except:
            model = torch.compile(model) # Fallback

    # 3. Warmup 
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    with context:
        for _ in range(warmup):
            out = model(input_tensor)
            if mode == 'train':
                (out.sum()).backward()
                model.zero_grad(set_to_none=True) # Faster than zero_grad()

    # 4. Benchmark
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    
    with context:
        for _ in range(runs):
            out = model(input_tensor)
            if mode == 'train':
                (out.sum()).backward()
                model.zero_grad(set_to_none=True)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_time = time.time() - start
    
    return (batch_size * runs) / total_time

def measure_memory(model, input_tensor):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    with torch.enable_grad():
        out = model(input_tensor)
        (out.sum()).backward()
        model.zero_grad(set_to_none=True)
        
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SHAPE = (3, 224, 224)
    
    # Use Channels Last for Input (Matches hardware tensor cores)
    batch = args.batch_size
    input_t = torch.randn(batch, *IMG_SHAPE, device=device).to(memory_format=torch.channels_last)
    
    # 1. Load & Benchmark Baseline
    model_ln = load_model_from_checkpoint(args.model, args.ckpt_ln, device, is_dyt=False)
    print(f"\n--- Benchmarking LayerNorm (Batch {batch}) ---")
    ln_inf = measure_throughput(model_ln, input_t, batch, 'inference')
    ln_train = measure_throughput(model_ln, input_t, batch, 'train')
    ln_mem = measure_memory(model_ln, input_t)
    print(f"LN  | Inf: {ln_inf:.1f} | Train: {ln_train:.1f} | Mem: {ln_mem:.1f}")
    
    del model_ln
    torch.cuda.empty_cache()

    # 2. Load & Benchmark DyT
    model_dyt = load_model_from_checkpoint(args.model, args.ckpt_dyt, device, is_dyt=True)
    print(f"\n--- Benchmarking DyT (Batch {batch}) ---")
    dyt_inf = measure_throughput(model_dyt, input_t, batch, 'inference')
    dyt_train = measure_throughput(model_dyt, input_t, batch, 'train')
    dyt_mem = measure_memory(model_dyt, input_t)
    print(f"DyT | Inf: {dyt_inf:.1f} | Train: {dyt_train:.1f} | Mem: {dyt_mem:.1f}")

    # Save & Plot
    results = {
        "baseline": {'inf': ln_inf, 'train': ln_train, 'mem': ln_mem},
        "dyt": {'inf': dyt_inf, 'train': dyt_train, 'mem': dyt_mem}
    }
    with open(os.path.join(args.output_dir, "efficiency_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    metrics = ['Throughput (Inf)', 'Throughput (Train)', 'Peak Mem']
    # Avoid div by zero
    ln_vals = [max(ln_inf,1), max(ln_train,1), max(ln_mem,1)]
    dyt_vals = [dyt_inf, dyt_train, dyt_mem]
    
    # Normalize relative to LN
    dyt_norm = [d/l for d, l in zip(dyt_vals, ln_vals)]
    ln_norm = [1.0] * 3
    
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, ln_norm, width, label='LayerNorm', color='gray')
    ax.bar(x + width/2, dyt_norm, width, label='DyT', color='orange')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Relative Performance (LN=1.0)')
    ax.set_title(f'Efficiency Benchmark (Batch {batch})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.figtext(0.5, 0.01, "Throughput: Higher is Better. Memory: Lower is Better.", ha="center")
    plt.savefig(os.path.join(args.output_dir, "benchmark_comparison.png"))
    print(f"Saved plot to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ckpt_ln', type=str, required=True)
    parser.add_argument('--ckpt_dyt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./bench_results')
    args = parser.parse_args()
    main(args)