import torch
import time
import numpy as np
import argparse
import os
import json
import timm
import matplotlib.pyplot as plt

def measure_throughput(model, input_tensor, batch_size, mode='inference', warmup=10, runs=50):
    model.eval() if mode == 'inference' else model.train()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for _ in range(warmup):
        out = model(input_tensor)
        if mode == 'train': (out.sum()).backward(); model.zero_grad()
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        out = model(input_tensor)
        if mode == 'train': (out.sum()).backward(); model.zero_grad()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    throughput = batch_size / ((time.time() - start) / runs)
    return throughput

def measure_memory(model, input_tensor):
    torch.cuda.reset_peak_memory_stats()
    model.train()
    (model(input_tensor).sum()).backward()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SHAPE = (3, 224, 224)
    
    # Load Models
    model_ln = timm.create_model(args.model, pretrained=False).to(device)
    # Assuming you have a way to create DyT model, e.g., explicit class or flag
    # For demo, we simulate DyT as the same model (replace with your dyt factory)
    model_dyt = timm.create_model(args.model, pretrained=False).to(device) 
    
    batch = args.batch_size
    input_t = torch.randn(batch, *IMG_SHAPE).to(device)
    
    print(f"Benchmarking Batch {batch}...")
    
    # Measure LN
    ln_res = {
        'inf': measure_throughput(model_ln, input_t, batch, 'inference'),
        'train': measure_throughput(model_ln, input_t, batch, 'train'),
        'mem': measure_memory(model_ln, input_t)
    }
    
    # Measure DyT
    dyt_res = {
        'inf': measure_throughput(model_dyt, input_t, batch, 'inference'),
        'train': measure_throughput(model_dyt, input_t, batch, 'train'),
        'mem': measure_memory(model_dyt, input_t)
    }
    
    # Plot
    metrics = ['Throughput (Inf)', 'Throughput (Train)', 'Peak Mem']
    ln_vals = [ln_res['inf'], ln_res['train'], ln_res['mem']]
    dyt_vals = [dyt_res['inf'], dyt_res['train'], dyt_res['mem']]
    
    # Normalize to LN = 1.0
    dyt_norm = [d/l for d, l in zip(dyt_vals, ln_vals)]
    ln_norm = [1.0] * 3
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, ln_norm, width, label='LayerNorm', color='gray')
    ax.bar(x + width/2, dyt_norm, width, label='DyT', color='orange')
    ax.set_ylabel('Relative Performance (Higher is better for speed)')
    ax.set_title(f'Efficiency Benchmark (Batch {batch})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.savefig(os.path.join(args.output_dir, "benchmark_comparison.png"))
    print(f"Benchmark saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./bench_results')
    args = parser.parse_args()
    main(args)