import torch
import random
import pandas as pd

class PerturbationManager:
    def __init__(self, perturbation_type, severity, channel_fraction=0.5, target_layer_idx=-1):
        self.perturbation_type = perturbation_type
        self.severity = severity
        self.channel_fraction = channel_fraction
        self.target_layer_idx = target_layer_idx
        self.hooks = []
        self.cached_masks = {} 

    def _get_fixed_mask(self, x, layer_id):
        if layer_id in self.cached_masks:
            return self.cached_masks[layer_id]
        
        B, N, C = x.shape
        num_pert = int(C * self.channel_fraction)
        mask = torch.zeros(C, device=x.device)
        perm = torch.randperm(C)
        idx = perm[:num_pert]
        mask[idx] = 1.0
        mask = mask.view(1, 1, -1)
        self.cached_masks[layer_id] = mask
        return mask

    def perturb_pre_hook(self, module, args):
        x = args[0]
        x_pert = x.clone()
        layer_id = module.layer_idx_tracker
        
        should_perturb = (self.target_layer_idx == -1) or (self.target_layer_idx == layer_id)
        mask = self._get_fixed_mask(x, layer_id)
        
        # --- Measure Clean Stats ---
        pert_indices = torch.where(mask.view(-1) == 1.0)[0]
        if len(pert_indices) > 0:
            clean_ch_means = x.mean(dim=(0, 1))[pert_indices].detach().cpu().numpy()
            clean_ch_stds = x.std(dim=(0, 1))[pert_indices].detach().cpu().numpy()
        else:
            clean_ch_means, clean_ch_stds = [], []

        # --- Apply Perturbation ---
        if should_perturb:
            if self.perturbation_type == 'scale':
                scale_tensor = mask * self.severity + (1 - mask)
                x_pert = x_pert * scale_tensor
            elif self.perturbation_type == 'bias':
                batch_std = x.std() # Approx scaling
                bias_val = self.severity * batch_std
                x_pert = x_pert + (mask * bias_val)
            elif self.perturbation_type == 'noise':
                noise = torch.randn_like(x) * x.std()
                x_pert = (1 - mask) * x + mask * noise

        # --- Measure Perturbed Stats ---
        if len(pert_indices) > 0:
            pert_ch_means = x_pert.mean(dim=(0, 1))[pert_indices].detach().cpu().numpy()
            pert_ch_stds = x_pert.std(dim=(0, 1))[pert_indices].detach().cpu().numpy()
        else:
            pert_ch_means, pert_ch_stds = [], []

        module.temp_stats = {
            'indices': pert_indices.cpu().numpy(),
            'is_perturbed': should_perturb,
            'clean_mean': clean_ch_means,
            'clean_std': clean_ch_stds,
            'pert_mean': pert_ch_means,
            'pert_std': pert_ch_stds
        }
        return (x_pert,)

    def collect_post_hook(self, module, input, output):
        if not hasattr(module, 'temp_stats'):
            return 

        temp = module.temp_stats
        pert_indices = temp['indices']
        
        if len(pert_indices) > 0:
            after_ch_means = output.mean(dim=(0, 1))[pert_indices].detach().cpu().numpy()
            after_ch_stds = output.std(dim=(0, 1))[pert_indices].detach().cpu().numpy()
            
            if not hasattr(module, 'perturbation_stats'):
                module.perturbation_stats = []
            
            for k, idx in enumerate(pert_indices):
                module.perturbation_stats.append({
                    'layer_idx': module.layer_idx_tracker,
                    'channel_idx': idx,
                    'is_perturbed': temp['is_perturbed'],
                    'clean_mean': temp['clean_mean'][k],
                    'pert_mean': temp['pert_mean'][k],
                    'mean_after': after_ch_means[k],
                    'clean_std': temp['clean_std'][k],
                    'pert_std': temp['pert_std'][k],
                    'std_after': after_ch_stds[k]
                })
        del module.temp_stats

    def register_hooks(self, model):
        count = 0
        for i, block in enumerate(model.blocks):
            target_layer = getattr(block, 'norm1', None)
            if target_layer is not None:
                target_layer.layer_idx_tracker = i
                h1 = target_layer.register_forward_pre_hook(self.perturb_pre_hook)
                self.hooks.append(h1)
                h2 = target_layer.register_forward_hook(self.collect_post_hook)
                self.hooks.append(h2)
                count += 1
        print(f"Registered Pre/Post hooks on {count} layers.")