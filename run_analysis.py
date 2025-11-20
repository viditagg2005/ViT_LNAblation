##############################################################
# This file runs the Information Preservation Analysis Suite #
##############################################################

import argparse
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# Import your Analysis Suite
from info_preservation_analysis import AnalysisSuite

def load_model(model_name, checkpoint_path, device):
    """Loads model architecture and weights."""
    # Create model structure (Use your repo's create_model if available)
    print(f"Creating model {model_name}...")
    model = timm.create_model(model_name, pretrained=False,num_classes = 100)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only = False)
        # Handle 'model' key if present (common in timm/deit)
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
    """Creates a CIFAR-100 Validation Loader."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Check if path exists, else download to ./data
    if not os.path.exists(data_path):
        print("Data path not found, downloading to ./data")
        data_path = "./data"
        
    dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Data
    print("Setting up data...")
    dataloader = get_dataloader(args.data_path)
    
    # 2. Load Models
    model_ln = load_model(args.model, args.ckpt_ln, device)
    model_dyt = load_model(args.model, args.ckpt_dyt, device)
    
    suite = AnalysisSuite(device=device)
    results = {}

    # --- Experiment A: Effective Rank ---
    print("\n[1/2] Computing Effective Rank...")
    ranks_ln, names_ln = suite.compute_rank_profile(model_ln, dataloader)
    ranks_dyt, names_dyt = suite.compute_rank_profile(model_dyt, dataloader)
    
    # Plot Rank
    plt.figure(figsize=(10, 6))
    plt.plot(ranks_ln, label='LayerNorm (Baseline)', marker='o')
    plt.plot(ranks_dyt, label='DynamicTanh (DyT)', marker='x')
    plt.xlabel('Layer Depth')
    plt.ylabel('Effective Rank')
    plt.title(f'Representation Collapse Analysis: {args.model}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    rank_plot_path = os.path.join(args.output_dir, "rank_comparison.png")
    plt.savefig(rank_plot_path)
    print(f"Saved rank plot to {rank_plot_path}")
    
    # Save Rank Data
    results['effective_rank'] = {
        'layers': names_ln,
        'ln_scores': ranks_ln,
        'dyt_scores': ranks_dyt
    }

    # --- Experiment B: CKA Similarity ---
    print("\n[2/2] Computing CKA Similarity Matrix...")
    # This compares layers of LN model vs DyT model
    l1, l2, matrix = suite.compute_cka(model_ln, model_dyt, dataloader)
    
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap="magma")
    plt.title("CKA: DyT Layers (Y) vs LN Layers (X)")
    plt.xlabel("LayerNorm Model Depth")
    plt.ylabel("DyT Model Depth")
    cka_plot_path = os.path.join(args.output_dir, "cka_heatmap.png")
    plt.savefig(cka_plot_path)
    print(f"Saved CKA heatmap to {cka_plot_path}")
    
    # Save Results JSON
    json_path = os.path.join(args.output_dir, "analysis_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[SUCCESS] All analysis data saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--ckpt_ln', type=str, required=True, help='Path to Baseline/LN checkpoint')
    parser.add_argument('--ckpt_dyt', type=str, required=True, help='Path to DyT checkpoint')
    parser.add_argument('--output_dir', type=str, default='./analysis_output')
    args = parser.parse_args()
    
    main(args)