import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class AnalysisSuite:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_rank_profile(self, model, dataloader, layer_keyword="blocks"):
        """Computes Shannon Effective Rank per layer."""
        model.eval()
        ranks = {}
        
        def hook(name):
            def fn(m, i, o):
                # Flatten (B, T, C) -> (B*T, C)
                act = o.detach()
                if act.dim() == 3: act = act.reshape(-1, act.shape[-1])
                act = act - act.mean(dim=0)
                
                # SVD
                try:
                    _, S, _ = torch.linalg.svd(act.float(), full_matrices=False)
                    p = S / S.sum()
                    p = p[p > 0]
                    entropy = -torch.sum(p * torch.log(p))
                    ranks[name] = torch.exp(entropy).item()
                except RuntimeError:
                    ranks[name] = 0.0
            return fn

        hooks = []
        for name, m in model.named_modules():
            # Looks for blocks.0, blocks.1, etc.
            if layer_keyword in name and "." in name and isinstance(m, nn.Module):
                # Avoid sub-modules inside the block, grab the block output
                if len(name.split('.')) <= 2: 
                    hooks.append(m.register_forward_hook(hook(name)))
        
        inputs, _ = next(iter(dataloader))
        with torch.no_grad():
            model(inputs.to(self.device))
            
        for h in hooks: h.remove()
        
        # Return sorted keys and values
        keys = sorted(ranks.keys(), key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        return [ranks[k] for k in keys], keys

    def compute_cka(self, model1, model2, dataloader, layer_keyword="blocks"):
        """Computes CKA similarity matrix."""
        model1.eval(); model2.eval()
        acts1, acts2 = {}, {}

        def get_act(storage, name):
            def hook(m, i, o):
                flat_act = o.detach().flatten(1)
                storage[name] = flat_act
            return hook

        # Register hooks
        hooks = []
        for model, storage in [(model1, acts1), (model2, acts2)]:
            for name, m in model.named_modules():
                if layer_keyword in name and "." in name and isinstance(m, nn.Module):
                     if len(name.split('.')) <= 2:
                        hooks.append(m.register_forward_hook(get_act(storage, name)))

        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(self.device)
        with torch.no_grad():
            model1(inputs)
            model2(inputs)

        for h in hooks: h.remove()

        l1_names = sorted(acts1.keys(), key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        l2_names = sorted(acts2.keys(), key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        matrix = np.zeros((len(l1_names), len(l2_names)))

        print("[CKA] Computing matrix...")
        for i, k1 in enumerate(tqdm(l1_names)):
            X = acts1[k1]
            for j, k2 in enumerate(l2_names):
                Y = acts2[k2]
                matrix[i, j] = self._linear_hsic(X, Y) / (np.sqrt(self._linear_hsic(X, X) * self._linear_hsic(Y, Y)))
        
        return l1_names, l2_names, matrix

    def _linear_hsic(self, X, Y):
        X, Y = X.float(), Y.float()
        KX = torch.mm(X, X.T)
        KY = torch.mm(Y, Y.T)
        n = KX.shape[0]
        H = torch.eye(n, device=self.device) - (1.0/n) * torch.ones((n, n), device=self.device)
        KH_X = torch.mm(KX, H)
        KH_Y = torch.mm(KY, H)
        return torch.trace(torch.mm(KH_X, KH_Y)).item()