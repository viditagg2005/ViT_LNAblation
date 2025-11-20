import torch
import time
import numpy as np
import argparse
import os
import json
from torch.profiler import profile, record_function, ProfilerActivity
import timm

# NOTE: If you have a custom model factory, import it here:
# from models import create_model 

def measure_throughput(model, input_tensor, batch_size, mode='inference', warmup=10, runs=50):
    model.eval() if mode == 'inference' else model.train()
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for _ in range(warmup):
        out = model(input_tensor)
        if mode == 'train':
            loss = out.sum()
            loss.backward()
            model.zero_grad()
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(runs):
        out = model(input_tensor)
        if mode == 'train':
            loss = out.sum()
            loss.backward()
            model.zero_grad()
            
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_run = total_time / runs
    throughput = batch_size / avg_time_per_run
    
    return throughput, avg_time_per_run * 1000 # img/sec, ms

def measure_memory(model, input_tensor):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.train()
    out = model(input_tensor)
    loss = out.sum()
    loss.backward()
    mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    return mem

def profile_kernels(model, input_tensor, output_dir):
    print("\n[INFO] Profiling Kernels... (Exporting chrome trace)")
    model.eval()
    trace_path = os.path.join(output_dir, "efficiency_trace.json")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(input_tensor)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to {trace_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--dynamic_tanh', action='store_true', help='Use DyT')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 64, 128])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='./efficiency_logs', help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Benchmarking {args.model} (DyT={args.dynamic_tanh}) on {device}...")

    # Create Model
    try:
        # If using standard timm
        model = timm.create_model(args.model, pretrained=False) 
        # If you have custom DyT logic, insert it here, e.g.:
        # if args.dynamic_tanh: apply_dyt(model)
    except:
        print("Error creating model. Ensure you are using the correct model name or factory.")
        exit()
        
    model.to(device)
    IMG_SHAPE = (3, args.image_size, args.image_size)
    
    # Dictionary to store all results for JSON dump
    full_results = {
        "model": args.model,
        "dynamic_tanh": args.dynamic_tanh,
        "device": device,
        "metrics": {}
    }

    print("-" * 70)
    print(f"{'Batch':<10} {'Mode':<10} {'Throughput (img/s)':<20} {'Latency (ms)':<15} {'Mem (MB)':<10}")
    print("-" * 70)

    for b in args.batch_sizes:
        input_t = torch.randn(b, *IMG_SHAPE).to(device)
        batch_res = {}
        
        # Inference
        inf_thr, inf_lat = measure_throughput(model, input_t, b, mode='inference')
        print(f"{b:<10} {'Infer':<10} {inf_thr:<20.2f} {inf_lat:<15.2f} {'-':<10}")
        batch_res['inference_throughput'] = inf_thr
        batch_res['inference_latency'] = inf_lat
        
        # Train
        try:
            train_thr, train_lat = measure_throughput(model, input_t, b, mode='train')
            mem = measure_memory(model, input_t)
            print(f"{b:<10} {'Train':<10} {train_thr:<20.2f} {train_lat:<15.2f} {mem:<10.2f}")
            batch_res['train_throughput'] = train_thr
            batch_res['train_latency'] = train_lat
            batch_res['peak_memory_mb'] = mem
        except RuntimeError as e:
            print(f"{b:<10} {'Train':<10} OOM or Error: {e}")
            batch_res['error'] = str(e)

        full_results["metrics"][f"batch_{b}"] = batch_res

    # Save JSON
    json_path = os.path.join(args.output_dir, "efficiency_metrics.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=4)
    print(f"\n[SUCCESS] Metrics saved to {json_path}")

    # Run Profiler on Batch 32
    if device == 'cuda':
        dummy = torch.randn(32, *IMG_SHAPE).to(device)
        profile_kernels(model, dummy, args.output_dir)