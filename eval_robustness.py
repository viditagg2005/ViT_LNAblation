import argparse
import torch
import pandas as pd
import utils
from datasets import build_dataset
from timm.models import create_model
from dynamic_tanh import convert_ln_to_dyt 
from robustness_hooks import PerturbationManager

# --- Fix 1: Define Boolean Parsing Locally ---
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# --- Fix 2: Define Accuracy Function Locally ---
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def get_args():
    parser = argparse.ArgumentParser('DyT Robustness Evaluation')
    
    # --- Model & Data ---
    parser.add_argument('--model', default='vit_tiny_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    parser.add_argument('--dynamic_tanh', type=str2bool, default=False)
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT', help='Drop path rate')
    
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--data_set', default='CIFAR', choices=['CIFAR', 'IMNET', 'image_folder'])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)
    
    # --- Transforms (Required by dataset.py) ---
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--crop_pct', type=float, default=None)

    # --- Perturbation Control ---
    parser.add_argument('--perturb_type', default='none', choices=['none', 'scale', 'bias', 'noise'])
    parser.add_argument('--severity', type=float, default=1.0)
    parser.add_argument('--perturb_layer', type=int, default=-1, help='Layer index to perturb (-1 for all)')
    parser.add_argument('--channel_fraction', type=float, default=0.5, help='Fraction of channels to perturb')
    
    return parser.parse_args()

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    all_channel_stats = []

    for batch_idx, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        
        # FIX: Call local accuracy function instead of utils.accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
        metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))
        metric_logger.meters['loss'].update(loss.item(), n=images.size(0))

        # --- Extract Stats from Hooks ---
        for i, block in enumerate(model.blocks):
            norm_layer = getattr(block, 'norm1', None)
            if hasattr(norm_layer, 'perturbation_stats') and len(norm_layer.perturbation_stats) > 0:
                all_channel_stats.extend(norm_layer.perturbation_stats)
                norm_layer.perturbation_stats = []

    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f}')
    return metric_logger.acc1.global_avg, all_channel_stats

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Setup
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, 
        pin_memory=args.pin_mem, drop_last=False
    )

    # 2. Model Creation
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=nb_classes,
        global_pool='avg',
        drop_path_rate=args.drop_path,
    )
    
    if args.dynamic_tanh:
        print("Converting LayerNorm to DynamicTanh...")
        model = convert_ln_to_dyt(model)

    model.to(device)

    # 3. Load Checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
        if 'head.weight' in state_dict and state_dict['head.weight'].shape[0] != nb_classes:
            print(f"Removing head from checkpoint (classes mismatch)")
            del state_dict['head.weight']
            del state_dict['head.bias']
            
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {msg}")
    else:
        print("WARNING: No checkpoint loaded. Testing random weights.")

    # 4. Attach Hooks
    if args.perturb_type != 'none':
        print(f"Applying {args.perturb_type} | Severity: {args.severity} | Layer: {args.perturb_layer}")
        manager = PerturbationManager(
            args.perturb_type, 
            args.severity,
            channel_fraction=args.channel_fraction,
            target_layer_idx=args.perturb_layer
        )
        manager.register_hooks(model)
    
    # 5. Evaluate
    acc, raw_stats = evaluate(data_loader_val, model, device)
    
    # 6. Save Results
    results_fname = f"stats_{args.perturb_type}_{args.severity}_L{args.perturb_layer}.csv"
    
    if raw_stats:
        print(f"Aggregating {len(raw_stats)} stats entries...")
        df = pd.DataFrame(raw_stats)
        
        cols = [
            'layer_idx', 'channel_idx', 'is_perturbed',
            'clean_mean', 'pert_mean', 'mean_after',
            'clean_std', 'pert_std', 'std_after'
        ]
        # Safety check
        present_cols = [c for c in cols if c in df.columns]
        df = df[present_cols]
        
        df_avg = df.groupby(['layer_idx', 'channel_idx', 'is_perturbed']).mean().reset_index()
        
        df_avg['global_acc1'] = acc
        df_avg.to_csv(results_fname, index=False)
        print(f"Detailed stats saved to {results_fname}")
    else:
        with open(results_fname, 'w') as f:
            f.write(f"global_acc1,{acc}\n")

if __name__ == '__main__':
    main()