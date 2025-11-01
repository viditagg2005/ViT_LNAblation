import torch
import torch.nn as nn
import numpy as np
from timm.layers import LayerNorm2d





### Helper Functions for Instrumentation ###
def _flatten_activations(x: torch.Tensor) -> np.ndarray:
    """
    Flatten activations x into a 2D numpy array of shape (N, C),
    where N is number of examples×tokens and C is feature dimension.
    For e.g. x of shape (B, T, C) → (B*T, C),
    or (B, C, H, W) → (B*H*W, C).
    """
    x_cpu = x.detach().cpu()
    if x_cpu.ndim == 3:
        B, T, C = x_cpu.shape
        return x_cpu.view(B * T, C).numpy()
    if x_cpu.ndim == 4:
        B, C, H, W = x_cpu.shape
        # move channels last then flatten
        return x_cpu.permute(0, 2, 3, 1).reshape(B * H * W, C).numpy()
    # fallback: collapse all but last dim
    return x_cpu.reshape(-1, x_cpu.shape[-1]).numpy()

def _basic_moments_np(arr: np.ndarray) -> dict:
    """
    Compute mean, std, skewness, kurtosis of a numpy array arr.
    """
    flat = arr.ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    if std < 1e-12:
        skew = 0.0
        kurt = 0.0
    else:
        m3 = np.mean((flat - mean) ** 3)
        m4 = np.mean((flat - mean) ** 4)
        skew = float(m3 / (std ** 3 + 1e-12))
        kurt = float(m4 / (std ** 4 + 1e-12) - 3.0)
    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}

def _randomized_svd(X: np.ndarray, k: int = 64, n_oversamples: int = 8, n_iter: int = 2, seed: int = None) -> tuple:
    """
    Approximate truncated SVD of X (shape n_samples × n_features) using randomized algorithm.
    Returns (U, S, Vt) where S is length ≤ k.
    Adapted from standard randomized SVD methods. :contentReference[oaicite:0]{index=0}
    """
    n, d = X.shape
    k = min(k, n, d)
    l = k + n_oversamples
    rng = np.random.RandomState(seed)
    # random Gaussian test matrix
    Omega = rng.normal(size=(d, l)).astype(X.dtype)
    Y = X @ Omega      # shape (n, l)
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ X        # shape (l, d)
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]
    return U[:, :k], S[:k], Vt[:k, :]

def _effective_rank_from_svals(svals: np.ndarray) -> float:
    """
    Compute Shannon‐entropy‐based effective rank from singular values array svals:
        eff_rank = exp(- sum(p_i * log(p_i)) ), where p_i = s_i / sum(s_i)
    """
    s = np.maximum(svals, 1e-12)
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))



class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32) * float(alpha_init_value))
        # instrumentation store for diagnostics
        self._instrumentation = {
            "alpha_snapshots": [],
            "grad_norms": {"in": [], "out": [], "ratio": []},
            "activation_stats": []
        }
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha.to(x.dtype) * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"
    

    
    ### Instrumentation and Analysis Methods ###
    
    def snapshot_alpha(self) -> dict:
        """Record summary statistics of alpha for logging."""
        with torch.no_grad():
            a_np = self.alpha.detach().cpu().numpy().ravel()
            summary = {"mean": float(a_np.mean()), "std": float(a_np.std()), "shape": tuple(a_np.shape)}
        self._instrumentation["alpha_snapshots"].append(summary)
        return summary

    def attach_grad_monitor(self, name: str, container: dict):
        """
        Attach hooks to monitor input/output gradient norms and ratio for this module.
        container[name] will be filled with lists of norms and ratio values.
        """
        container.setdefault(name, {"in_norm": [], "out_norm": [], "ratio": []})

        def _hook_in(grad):
            norm = float(grad.detach().cpu().norm().item())
            container[name]["in_norm"].append(norm)

        def _hook_out(grad):
            norm = float(grad.detach().cpu().norm().item())
            container[name]["out_norm"].append(norm)
            ins = container[name]["in_norm"][-1] if container[name]["in_norm"] else None
            ratio = float(norm / (ins + 1e-12)) if ins is not None else None
            container[name]["ratio"].append(ratio)

        def _forward_hook(mod, inp, out):
            if isinstance(inp, (tuple, list)) and isinstance(inp[0], torch.Tensor):
                inp[0].register_hook(_hook_in)
            if isinstance(out, torch.Tensor):
                out.register_hook(_hook_out)

        handle = self.register_forward_hook(_forward_hook)
        container[name]["_hook_handle"] = handle

    def remove_grad_monitor(self, name: str, container: dict):
        info = container.get(name, {})
        handle = info.get("_hook_handle", None)
        if handle is not None:
            handle.remove()
            info.pop("_hook_handle", None)

    def compute_activation_stats(self, x: torch.Tensor) -> dict:
        """
        Compute activation statistics (mean/std/skew/kurtosis) and estimate effective rank via randomized SVD.
        """
        x_np = _flatten_activations(x)
        stats = _basic_moments_np(x_np)
        try:
            U, S, Vt = _randomized_svd(x_np, k=min(64, x_np.shape[1]))
            eff = _effective_rank_from_svals(S)
            stats["top_singular"] = S[:5].tolist()
            stats["effective_rank"] = eff
        except Exception as e:
            stats["top_singular"] = None
            stats["effective_rank"] = None
        self._instrumentation["activation_stats"].append(stats)
        return stats

    def param_and_flops(self, sample_shape: tuple = None) -> dict:
        """
        Estimate extra parameters and approximate FLOPs for this module.
        sample_shape : (B, T, C) to compute total FLOPs for that batch.
        """
        extra_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        flops_per_elem = 2  # approximate: multiply + tanh
        if sample_shape is None:
            total_flops = None
        else:
            B, T, C = sample_shape
            total_flops = B * T * C * flops_per_elem
        return {"extra_params": int(extra_params),
                "flops_per_element": flops_per_elem,
                "flops_total": total_flops}


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

