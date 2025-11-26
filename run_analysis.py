import argparse
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timm

# Import Analysis Suite
from info_preservation_analysis import AnalysisSuite

# --- 1. Import Conversion Function ---
try:
    from dynamic_tanh_new import convert_ln_to_dyt
except ImportError:
    print("[ERROR] Could not import 'convert_ln_to_dyt' from 'dynamic_tanh_new.py'.")
    exit(1)

# --- 2. Robust Model Loading ---
def load_model(model_name, ckpt_path, device, is_dyt=False):
    print(f"Creating model {model_name} [{'DyT' if is_dyt else 'LN'}]...")
    
    # Create Standard Model
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=100)
    except:
        print(f"Warning: {model_name} creation failed with num_classes=100, trying default.")
        model = timm.create_model(model_name, pretrained=False) 

    # Convert to DyT if requested
    if is_dyt:
        print(" -> Converting LayerNorm to DynamicTanh...")
        model = convert_ln_to_dyt(model)

    model.to(device)
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # --- Fix Key Mismatches (Strict Prefix Check) ---
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            
            # Scenario A: Checkpoint has 'fc_norm...', Model expects 'norm...'
            if k.startswith('fc_norm') and hasattr(model, 'norm'):
                new_k = k.replace('fc_norm', 'norm', 1)
                
            # Scenario B: Checkpoint has 'norm...', Model expects 'fc_norm...'
            # We explicitly check that it DOES NOT start with 'fc_norm' to avoid fc_fc_norm
            elif k.startswith('norm') and not k.startswith('fc_norm') and hasattr(model, 'fc_norm'):
                new_k = k.replace('norm', 'fc_norm', 1)
            
            new_state_dict[new_k] = v
            
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load status: {msg}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found! Using random init.")
    
    return model

# --- 3. Data Loading ---
def get_dataloader(data_path, batch_size=64):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if not os.path.exists(data_path): os.makedirs(data_path, exist_ok=True)
    dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- 4. Plotting Logic ---
def parse_training_logs(log_path, model_type):
    data = []
    if not os.path.exists(log_path): 
        print(f"Log not found: {log_path}")
        return []
    
    with open(log_path, 'r') as f:
        for line in f:
            try: 
                entry = json.loads(line)
                epoch = entry['epoch']
                step = entry['step']
                abs_step = epoch * 1000 + step 
                
                for mod_name, metrics in entry['modules'].items():
                    row = {
                        'epoch': epoch, 
                        'step': abs_step, 
                        'module': mod_name,
                        'model_type': model_type
                    }
                    row.update(metrics)
                    data.append(row)
            except: pass
    return data

def plot_training_dynamics(log_ln, log_dyt, output_dir):
    print("\nGenerating Dynamics Plots...")
    
    records = []
    if log_ln: records.extend(log_ln)
    if log_dyt: records.extend(log_dyt)
    
    if not records:
        print("No valid log data found to plot.")
        return

    df = pd.DataFrame(records)
    
    # Filter for cleaner plots (only main blocks)
    df['layer_idx'] = df['module'].apply(lambda x: x.split('.')[1] if 'blocks' in x else 'other')
    df_blocks = df[df['layer_idx'] != 'other']

    # --- Plot A: Gradient Ratio (Stability Comparison) ---
    if 'grad_ratio' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_blocks, 
            x='epoch', 
            y='grad_ratio', 
            hue='model_type', 
            style='model_type',
            palette={'LayerNorm': 'blue', 'DyT': 'orange'}
        )
        plt.title("Average Gradient Norm Ratio (Out/In) vs Epoch")
        plt.ylabel("Gradient Ratio (Ideal ~ 1.0)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "dyn_grad_stability_compare.png"))
        plt.close()
        print("Saved dyn_grad_stability_compare.png")

    # --- Plot B: Alpha Evolution (DyT Only) ---
    if 'alpha' in df.columns:
        df_dyt = df_blocks[df_blocks['model_type'] == 'DyT']
        if not df_dyt.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df_dyt, x='epoch', y='alpha', hue='layer_idx', palette='coolwarm')
            plt.title("DyT Alpha Parameter Evolution per Layer")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "dyn_alpha_evolution.png"))
            plt.close()
            print("Saved dyn_alpha_evolution.png")

    # --- Plot C: Gradient Norms (Log Scale) ---
    if 'grad_in' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_blocks, 
            x='epoch', 
            y='grad_in', 
            hue='model_type',
            palette={'LayerNorm': 'blue', 'DyT': 'orange'}
        )
        plt.title("Average Gradient Norm (Input) vs Epoch")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "dyn_grad_norms.png"))
        plt.close()
        print("Saved dyn_grad_norms.png")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Data
    dataloader = get_dataloader(args.data_path)
    
    # 2. Load Models (Pass is_dyt flag!)
    model_ln = load_model(args.model, args.ckpt_ln, device, is_dyt=False)
    model_dyt = load_model(args.model, args.ckpt_dyt, device, is_dyt=True)
    
    suite = AnalysisSuite(device=device)

    # --- Phase 1: Effective Rank ---
    print("\n[1/3] Computing Final Effective Rank Profile...")
    ranks_ln, names_ln = suite.compute_rank_profile(model_ln, dataloader)
    ranks_dyt, names_dyt = suite.compute_rank_profile(model_dyt, dataloader)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks_ln, label='LayerNorm', marker='o')
    plt.plot(ranks_dyt, label='DyT', marker='x')
    plt.xlabel('Layer Depth')
    plt.ylabel('Effective Rank')
    plt.title('Final Representation Rank (Information Preservation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "rank_comparison.png"))
    plt.close()

    # --- Phase 2: CKA ---
    print("\n[2/3] Computing CKA Similarity...")
    l1, l2, matrix = suite.compute_cka(model_ln, model_dyt, dataloader)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap="magma")
    plt.title("CKA: DyT Layers (Y) vs LN Layers (X)")
    plt.savefig(os.path.join(args.output_dir, "cka_heatmap.png"))
    plt.close()
    
    # --- Phase 3: Training Dynamics ---
    print("\n[3/3] Parsing Training Logs...")
    path_ln_log = os.path.join(os.path.dirname(args.ckpt_ln), "training_log.json")
    path_dyt_log = os.path.join(os.path.dirname(args.ckpt_dyt), "training_log.json")
    
    log_data_ln = parse_training_logs(path_ln_log, "LayerNorm")
    log_data_dyt = parse_training_logs(path_dyt_log, "DyT")
    
    plot_training_dynamics(log_data_ln, log_data_dyt, args.output_dir)

    print(f"\n[SUCCESS] All analysis outputs saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--ckpt_ln', type=str, required=True, help="Path to LN checkpoint")
    parser.add_argument('--ckpt_dyt', type=str, required=True, help="Path to DyT checkpoint")
    parser.add_argument('--output_dir', type=str, default='./analysis_output')
    args = parser.parse_args()
    main(args)