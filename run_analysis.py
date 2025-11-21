import argparse
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timm

# Import your Analysis Suite class
from info_preservation_analysis import AnalysisSuite

# --- Loaders (Same as before) ---
def load_model(model_name, checkpoint_path, device):
    print(f"Creating model {model_name}...")
    model = timm.create_model(model_name, pretrained=False, num_classes=100)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        # weights_only=False to support numpy scalars often found in checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load status: {msg}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found! Using random init.")
    return model.to(device)

def get_dataloader(data_path, batch_size=64):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Plotting Logic ---
def parse_training_logs(log_path):
    data = []
    if not os.path.exists(log_path): return []
    with open(log_path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: pass
    return data

def plot_training_dynamics(log_data, output_dir):
    print("Generating Dynamics Plots (Gradient Flow & Alpha)...")
    if not log_data:
        print("No log data found.")
        return

    # Parse JSONL to DataFrame
    records = []
    for entry in log_data:
        epoch = entry['epoch']
        step = entry['step']
        abs_step = epoch * 1000 + step 
        
        for mod_name, metrics in entry['modules'].items():
            row = {'epoch': epoch, 'step': abs_step, 'module': mod_name}
            row.update(metrics)
            records.append(row)
            
    df = pd.DataFrame(records)
    if df.empty: return

    # Helper: shorter names for legend
    df['layer_idx'] = df['module'].apply(lambda x: x.split('.')[1] if 'blocks' in x else 'other')
    df_blocks = df[df['layer_idx'] != 'other']

    # 1. Gradient Ratio (Stability)
    if 'grad_ratio' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_blocks, x='step', y='grad_ratio', hue='layer_idx', palette='viridis', legend=False)
        plt.title("Gradient Norm Ratio (Out/In) vs Steps")
        plt.ylim(0, 3) # Zoom in on stable region
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dyn_grad_ratio.png"))
        plt.close()

    # 2. Alpha Evolution (Topology)
    if 'alpha' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_blocks, x='epoch', y='alpha', hue='layer_idx', palette='coolwarm')
        plt.title("Alpha Parameter Evolution per Module")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dyn_alpha.png"))
        plt.close()

    # 3. Gradient Norms (Raw)
    if 'grad_in' in df.columns and 'grad_out' in df.columns:
        plt.figure(figsize=(12, 6))
        # Plotting mean grad_in over time
        sns.lineplot(data=df_blocks, x='epoch', y='grad_in', hue='layer_idx', palette='magma', legend=False)
        plt.title("Gradient In-Norm vs Epoch")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dyn_grad_norms.png"))
        plt.close()

    print(f"Dynamics plots saved to {output_dir}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Data & Models
    dataloader = get_dataloader(args.data_path)
    # We pass args.model (e.g., 'vit_tiny_patch16_224') so timm knows what to build.
    print("Loading Baseline Model...")
    model_ln = load_model(args.model, args.ckpt_ln, device)

    print("Loading DyT Model...")
    model_dyt = load_model(args.model, args.ckpt_dyt, device)
    
    suite = AnalysisSuite(device=device)
    results = {}

    # --- Phase A: Static Rank Analysis (Computed ONCE on final weights) ---
    print("\n[1/2] Computing Final Effective Rank Profile...")
    ranks_ln, names_ln = suite.compute_rank_profile(model_ln, dataloader)
    ranks_dyt, names_dyt = suite.compute_rank_profile(model_dyt, dataloader)
    
    # Plot Rank Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(ranks_ln, label='LayerNorm', marker='o')
    plt.plot(ranks_dyt, label='DyT', marker='x')
    plt.xlabel('Layer Depth')
    plt.ylabel('Effective Rank')
    plt.title('Final Representation Rank (Higher = More Info)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "rank_comparison.png"))
    plt.close()

    # --- Phase B: CKA Analysis ---
    print("\n[2/2] Computing CKA Similarity...")
    l1, l2, matrix = suite.compute_cka(model_ln, model_dyt, dataloader)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap="magma")
    plt.title("CKA: DyT Layers (Y) vs LN Layers (X)")
    plt.savefig(os.path.join(args.output_dir, "cka_heatmap.png"))
    plt.close()
    
    # --- Phase C: Plot Training Dynamics (From Logs) ---
    # We look for training_log.json in the folder where the checkpoint came from
    log_path = os.path.join(os.path.dirname(args.ckpt_dyt), "training_log.json")
    if os.path.exists(log_path):
        log_data = parse_training_logs(log_path)
        plot_training_dynamics(log_data, args.output_dir)
    else:
        print(f"No training log found at {log_path}, skipping dynamics plots.")

    print(f"\n[SUCCESS] All analysis outputs saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--ckpt_ln', type=str, required=True, help='Path to Baseline checkpoint')
    parser.add_argument('--ckpt_dyt', type=str, required=True, help='Path to DyT checkpoint')
    parser.add_argument('--output_dir', type=str, default='./analysis_output')
    args = parser.parse_args()
    
    main(args)