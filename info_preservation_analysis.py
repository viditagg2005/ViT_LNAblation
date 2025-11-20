import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AnalysisSuite:
    def __init__(self, device='cuda'):
        self.device = device

    # ==============================
    # 1. CKA (Centered Kernel Alignment)
    # ==============================
    def compute_cka(self, model1, model2, dataloader, layer_keyword="blocks"):
        """
        Computes CKA similarity matrix between two models.
        """
        model1.eval(); model2.eval()
        acts1, acts2 = {}, {}

        def get_act(storage, name):
            def hook(m, i, o):
                # Flatten (B, T, C) -> (B, T*C) for full representation comparison
                flat_act = o.detach().flatten(1)
                storage[name] = flat_act
            return hook

        # Register hooks
        hooks = []
        for model, storage in [(model1, acts1), (model2, acts2)]:
            for name, m in model.named_modules():
                if layer_keyword in name and "." in name and isinstance(m, nn.Module):
                    hooks.append(m.register_forward_hook(get_act(storage, name)))

        # Forward pass (One Batch)
        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(self.device)
        with torch.no_grad():
            model1(inputs)
            model2(inputs)

        # Remove hooks
        for h in hooks: h.remove()

        # Compute Matrix
        l1_names = sorted(acts1.keys())
        l2_names = sorted(acts2.keys())
        matrix = np.zeros((len(l1_names), len(l2_names)))

        print("[CKA] Computing similarity matrix...")
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
        # Centering
        n = KX.shape[0]
        H = torch.eye(n, device=self.device) - (1.0/n) * torch.ones((n, n), device=self.device)
        # HSIC = tr(KX * H * KY * H)
        KH_X = torch.mm(KX, H)
        KH_Y = torch.mm(KY, H)
        return torch.trace(torch.mm(KH_X, KH_Y)).item()

    # ==============================
    # 2. Effective Rank
    # ==============================
    def compute_rank_profile(self, model, dataloader, layer_keyword="blocks"):
        """
        Computes Shannon Effective Rank per layer.
        """
        model.eval()
        ranks = {}
        
        def hook(name):
            def fn(m, i, o):
                # Flatten (B, T, C) -> (B*T, C)
                act = o.detach()
                if act.dim() == 3: act = act.reshape(-1, act.shape[-1])
                
                # Center
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
            if layer_keyword in name and "." in name and isinstance(m, nn.Module):
                hooks.append(m.register_forward_hook(hook(name)))
        
        inputs, _ = next(iter(dataloader))
        with torch.no_grad():
            model(inputs.to(self.device))
            
        for h in hooks: h.remove()
        
        # Return sorted by layer name
        return [ranks[k] for k in sorted(ranks.keys())], sorted(ranks.keys())

    # ==============================
    # 3. Linear Probing
    # ==============================
    def run_linear_probe(self, model, dataloader, layer_names, num_classes=1000, steps=200):
        """
        Trains a linear classifier on top of frozen features for specific layers.
        """
        model.eval()
        results = {}
        
        for layer in layer_names:
            print(f"[Probe] Training on layer: {layer}")
            # 1. Feature Extraction (Simple approach: extract 1 batch of features)
            # For rigorous results, use a full feature-extraction pass. 
            # Here we do a quick probe on limited data for feasibility.
            features = []
            targets = []
            
            def hook(m, i, o):
                # Global Avg Pool for classification
                f = o.detach()
                if f.dim() == 3: f = f.mean(dim=1)
                features.append(f)
            
            # Attach hook
            module = dict(model.named_modules())[layer]
            h = module.register_forward_hook(hook)
            
            # Collect Data (e.g., 5 batches)
            for idx, (x, y) in enumerate(dataloader):
                with torch.no_grad(): model(x.to(self.device))
                targets.append(y.to(self.device))
                if idx >= 4: break # Limit data
            
            h.remove()
            
            X = torch.cat(features)
            Y = torch.cat(targets)
            
            # 2. Train Classifier
            probe = nn.Linear(X.shape[1], num_classes).to(self.device)
            opt = optim.Adam(probe.parameters(), lr=0.01)
            crit = nn.CrossEntropyLoss()
            
            for _ in range(steps):
                opt.zero_grad()
                out = probe(X)
                loss = crit(out, Y)
                loss.backward()
                opt.step()
            
            # Accuracy
            pred = probe(X).argmax(dim=1)
            acc = (pred == Y).float().mean().item()
            results[layer] = acc
            
        return results