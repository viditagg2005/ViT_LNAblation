"""
models/dynamic_tanh.py

Advanced Dynamic Tanh (DyT) module for the DyT paper repo experiments.

Features:
- alpha parameterizations: scalar, per_channel, per_token, per_token_channel, factorized low-rank
- safe mixed-precision handling (alpha stored in float32, cast to input dtype on forward)
- optional affine (gamma, beta) after tanh
- APIs for:
    - clamp_alpha_(), snapshot_alpha(), get_alpha_stats()
    - attach_gradient_monitor(name, container) -> registers hooks to record grad norms and gradient-flow ratio
    - compute_activation_stats(tensor) -> mean/std/skew/kurtosis
    - estimate_effective_rank(X, k=64) -> randomized SVD-based effective rank (approx)
    - param_and_flops() -> counts extra params and estimates DyT FLOPs per element
- hooks are light-weight and intended for diagnostic instrumentation; not required for training.

Usage (simple):
  from models.dynamic_tanh import DynamicTanh
  dy = DynamicTanh(hidden_dim=768, mode="per_channel", use_affine=True)
  x = dy(x)   # identical callsite to LayerNorm replacement

Author: ChatGPT (adapted to your DyT repo)
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings


# -------------------------
# Utility functions
# -------------------------
def _ensure_tensor_on_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    return torch.tensor(t, device=device)


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    # Convert (N, T, C) or (N, C, H, W) to 2D matrix (N*T, C) or (N*H*W, C)
    if x.ndim == 3:
        N, T, C = x.shape
        return x.reshape(N * T, C)
    if x.ndim == 4:
        N, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(N * H * W, C)
    # fallback: collapse all but last dim
    return x.reshape(-1, x.shape[-1])


def _compute_basic_moments(x: np.ndarray) -> Dict[str, float]:
    # x: 1D or 2D numpy (we compute stats over all entries)
    flat = x.ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    # skewness / kurtosis robust via scipy replacement (compute manual)
    if flat.size < 2:
        skew = 0.0
        kurt = 0.0
    else:
        m3 = np.mean((flat - mean) ** 3)
        m4 = np.mean((flat - mean) ** 4)
        skew = float(m3 / (std ** 3 + 1e-12))
        kurt = float(m4 / (std ** 4 + 1e-12) - 3.0)
    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


# -------------------------
# Randomized SVD for effective rank estimation
# -------------------------
def randomized_svd(
    X: np.ndarray, k: int = 64, n_oversamples: int = 8, n_iter: int = 2, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate SVD(X) -> U, S, Vt using randomized algorithm.
    X: (n_samples, n_features) (float32 / float64 numpy)
    k: target number singular values (max)
    Returns U (n x k), S (k,), Vt (k x d)
    """
    n, d = X.shape
    k = min(k, min(n, d))
    l = k + n_oversamples
    rng = np.random.RandomState(seed)
    # random gaussian test matrix
    Omega = rng.normal(size=(d, l)).astype(X.dtype)
    Y = X @ Omega  # (n, l)
    # power iterations to improve approx of spectrum
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    # orthonormal basis
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ X  # (l, d)
    # exact SVD on small matrix
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]
    return U[:, :k], S[:k], Vt[:k, :]


def effective_rank_shannon(singular_values: np.ndarray) -> float:
    s = singular_values.copy()
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


# -------------------------
# Main module
# -------------------------
class DynamicTanh(nn.Module):
    """
    Dynamic Tanh module with diagnostics and multiple alpha parameterizations.

    Args:
      hidden_dim: required for per_channel/factorized/per_token_channel modes (C)
      mode: one of {"scalar", "per_channel", "per_token", "per_token_channel", "factorized"}
      max_len: required for token-aware modes (max T)
      rank: for factorized mode (u: T x r, v: r x C)
      init_alpha: scalar initialization value
      use_affine: whether to add learned affine (gamma, beta) after tanh
    """

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        mode: str = "scalar",
        max_len: Optional[int] = None,
        rank: int = 8,
        init_alpha: float = 1.0,
        use_affine: bool = False,
        clamp_alpha: Tuple[float, float] = (1e-6, 1e6),
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.rank = rank
        self.init_alpha = float(init_alpha)
        self.use_affine = bool(use_affine)
        self.alpha_min, self.alpha_max = clamp_alpha

        # create alpha parameters (always stored in float32 for numeric stability)
        if mode == "scalar":
            self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))
        elif mode == "per_channel":
            assert hidden_dim is not None, "hidden_dim required for per_channel mode"
            self.alpha = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32) * float(init_alpha))
        elif mode == "per_token":
            assert max_len is not None, "max_len required for per_token"
            self.alpha = nn.Parameter(torch.ones(max_len, dtype=torch.float32) * float(init_alpha))
        elif mode == "per_token_channel":
            assert hidden_dim is not None and max_len is not None
            self.alpha = nn.Parameter(torch.ones(max_len, hidden_dim, dtype=torch.float32) * float(init_alpha))
        elif mode == "factorized":
            assert hidden_dim is not None and max_len is not None
            # initialize small-rank factors; scale by init_alpha in forward
            self.u = nn.Parameter(torch.randn(max_len, rank, dtype=torch.float32) * 0.01)
            self.v = nn.Parameter(torch.randn(rank, hidden_dim, dtype=torch.float32) * 0.01)
            # store scalar_init to scale product
            self.scalar_init = float(init_alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.use_affine:
            assert hidden_dim is not None
            self.gamma = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
            self.beta = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        # instrumentation containers (users can read these dicts during/after training)
        self.instrumentation = {
            "alpha_snapshots": [],  # list of dicts per snapshot
            "grad_stats": {},       # name->list of grads
            "activation_stats": {}, # name->list of activation stats
        }

    # ---------------------
    # Forward
    # ---------------------
    def forward(self, x: torch.Tensor, token_index: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Args:
          x: tensor with last-dim = features (e.g., B, T, C or B, C, H, W)
          token_index: optional (T,) long tensor when per_token mode used with varying positions
        """
        # keep alpha in float32 but cast to x.dtype just for arithmetic
        dtype = x.dtype
        device = x.device

        if self.mode == "scalar":
            a = self.alpha.to(dtype)
            y = torch.tanh(a * x)
        elif self.mode == "per_channel":
            C = x.shape[-1]
            a = self.alpha.to(dtype).view(*([1] * (x.ndim - 1)), -1)
            y = torch.tanh(a * x)
        elif self.mode == "per_token":
            # token axis is assumed to be dim=1 (B, T, C)
            assert x.ndim >= 3, "expected x (B,T,C) for per_token"
            T = x.shape[1]
            if token_index is None:
                a_slice = self.alpha[:T].to(dtype)
            else:
                # token_index: (T,) containing positions
                pos = token_index.to(device)
                a_slice = self.alpha[pos].to(dtype)
            a = a_slice.view(1, -1, 1)
            y = torch.tanh(a * x)
        elif self.mode == "per_token_channel":
            assert x.ndim >= 3
            T = x.shape[1]
            C = x.shape[-1]
            a_mat = self.alpha[:T, :C].to(dtype).view(1, T, C)
            y = torch.tanh(a_mat * x)
        elif self.mode == "factorized":
            assert x.ndim >= 3
            T = x.shape[1]
            C = x.shape[-1]
            U = self.u[:T, :]    # (T, r)
            V = self.v           # (r, C)
            alpha_mat = (U @ V) * self.scalar_init   # (T, C)
            a = alpha_mat.to(dtype).view(1, T, C)
            y = torch.tanh(a * x)
        else:
            raise RuntimeError("unexpected mode")

        if self.use_affine:
            g = self.gamma.to(dtype).view(*([1] * (y.ndim - 1)), -1)
            b = self.beta.to(dtype).view(*([1] * (y.ndim - 1)), -1)
            y = g * y + b
        return y

    # ---------------------
    # Utilities: clamp, snapshot alpha, stats, param/flops
    # ---------------------
    def clamp_alpha_(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        """In-place clamp alpha(s). For factorized mode this is a no-op."""
        if min_value is None:
            min_value = self.alpha_min
        if max_value is None:
            max_value = self.alpha_max
        if hasattr(self, "alpha"):
            with torch.no_grad():
                self.alpha.clamp_(min=min_value, max=max_value)

    def snapshot_alpha(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of alpha(s) and store in instrumentation."""
        data = {"mode": self.mode}
        if hasattr(self, "alpha"):
            data["alpha"] = self.alpha.detach().cpu().tolist()
            data["shape"] = tuple(self.alpha.shape)
        if hasattr(self, "u") and hasattr(self, "v"):
            data["u_mean"] = float(self.u.detach().cpu().mean())
            data["v_mean"] = float(self.v.detach().cpu().mean())
            data["u_shape"] = tuple(self.u.shape)
            data["v_shape"] = tuple(self.v.shape)
        if self.use_affine:
            data["gamma_mean"] = float(self.gamma.detach().cpu().mean())
            data["beta_mean"] = float(self.beta.detach().cpu().mean())
        self.instrumentation["alpha_snapshots"].append(data)
        return data

    def get_alpha_stats(self) -> Dict[str, Any]:
        """Return basic moments for alpha parameters (useful for plotting)."""
        if hasattr(self, "alpha"):
            a = self.alpha.detach().cpu().numpy().ravel()
            return _compute_basic_moments(a)
        elif hasattr(self, "u") and hasattr(self, "v"):
            # inspect product mean/std (cheap)
            with torch.no_grad():
                alpha_mat = (self.u @ self.v).detach().cpu().numpy()
            return _compute_basic_moments(alpha_mat)
        else:
            return {}

    def param_and_flops(self, sample_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Return estimate of extra parameters and elementwise FLOPs for DyT
        sample_shape: optional (B, T, C) to estimate per-batch FLOPs; if None we provide per-element ops.
        Note: tanh is ~4 FLOPs (cheap activation) + multiply 1 flop => we estimate 4 mul/add + 1 tanh ~ 6 ops per element.
        This is a rough device-agnostic proxy.
        """
        extra_params = 0
        if hasattr(self, "alpha"):
            extra_params += int(self.alpha.numel())
        if hasattr(self, "u") and hasattr(self, "v"):
            extra_params += int(self.u.numel() + self.v.numel())
        if self.use_affine:
            extra_params += int(self.gamma.numel() + self.beta.numel())
        # flops per element (approx)
        flops_per_element = 6  # multiply + tanh overhead approximate
        if sample_shape is not None:
            B, T, C = sample_shape
            total_elements = B * T * C
            total_flops = int(total_elements * flops_per_element)
            return {"extra_params": extra_params, "flops_total": total_flops, "flops_per_element": flops_per_element}
        else:
            return {"extra_params": extra_params, "flops_per_element": flops_per_element}

    # ---------------------
    # Activation / info diagnostics (use for representation analysis)
    # ---------------------
    def compute_activation_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Compute basic statistics over activation tensor x (CPU numpy).
        Also compute a small randomized SVD effective-rank estimate if the tensor is > 2D.
        Returns dict with mean/std/skew/kurtosis and optional eff_rank and top singular values.
        """
        x_cpu = x.detach().cpu().numpy()
        stats = _compute_basic_moments(x_cpu)
        try:
            # if multi-dim, flatten tokens->rows and compute randomized SVD
            X2d = _flatten_tokens(torch.from_numpy(x_cpu)).numpy()
            # run small randomized svd
            k = min(64, min(X2d.shape[0], X2d.shape[1]))
            if k <= 0:
                return stats
            _, S, _ = randomized_svd(X2d, k=k, n_oversamples=8, n_iter=2)
            stats["top_singular_values"] = S[:10].tolist()
            stats["effective_rank"] = effective_rank_shannon(S)
        except Exception as e:
            # If SVD fails due to memory, return moments only
            warnings.warn(f"effective-rank estimation failed: {e}")
        return stats

    # ---------------------
    # Gradient flow diagnostics
    # ---------------------
    def attach_gradient_monitor(self, module_name: str, container: Dict[str, Any]) -> None:
        """
        Register hooks on this module to track gradient norms of its input and output.
        container: a dict-like object where we append metrics (user-managed)
        Example:
            grad_container = {}
            module.attach_gradient_monitor("block.3.dyt", grad_container)
        After a backward, call grad_container['<name>'] which will have entries:
            {'input_grad_norms': [...], 'output_grad_norms': [...], 'grad_ratio': [...]}
        Note: these hooks add very small runtime cost and are intended for diagnostics.
        """
        name = module_name

        container.setdefault(name, {"input_grad_norms": [], "output_grad_norms": [], "grad_ratio": []})

        def _save_input_grad(grad):
            gn = float(grad.detach().cpu().norm().item())
            container[name]["input_grad_norms"].append(gn)

        def _save_output_grad(grad):
            gn = float(grad.detach().cpu().norm().item())
            container[name]["output_grad_norms"].append(gn)

        # register hooks: on forward we attach backward hooks to the tensors
        def _forward_hook(module, inp, out):
            # inp: tuple; out: tensor
            if isinstance(out, torch.Tensor):
                out.requires_grad_(True)
                out.register_hook(_save_output_grad)
            # for inputs we assume inp[0] is primary
            if isinstance(inp, (tuple, list)) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                in_t = inp[0]
                in_t.requires_grad_(True)
                in_t.register_hook(_save_input_grad)

        # store forward hook handle for potential removal by user
        h = self.register_forward_hook(_forward_hook)
        # store handle in container so caller can remove later if desired
        container[name]["_hook_handle"] = h

    def remove_gradient_monitor(self, module_name: str, container: Dict[str, Any]) -> None:
        """Remove hook handle if present."""
        info = container.get(module_name, None)
        if info is None:
            return
        h = info.get("_hook_handle", None)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
            info.pop("_hook_handle", None)

    # ---------------------
    # Local Lipschitz / Jacobian slope estimator for tanh( a * x )
    # ---------------------
    def local_slope_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-element derivative statistics for tanh(a*x):
            d/dx tanh(a x) = a * (1 - tanh(a x)^2)
        Return max_abs_slope, mean_abs_slope, std_abs_slope as floats.
        This is a local Lipschitz proxy (diagonal Jacobian for elementwise op).
        """
        with torch.no_grad():
            dtype = x.dtype
            if self.mode in ("scalar", "per_channel"):
                if self.mode == "scalar":
                    a = self.alpha.to(dtype)
                    a_b = float(a.item())
                    t = torch.tanh(a * x)
                    slopes = (a_b * (1.0 - t * t)).abs()
                else:  # per_channel
                    a = self.alpha.to(dtype).view(*([1] * (x.ndim - 1)), -1)
                    t = torch.tanh(a * x)
                    slopes = (a * (1.0 - t * t)).abs()
            elif self.mode in ("per_token", "per_token_channel", "factorized"):
                # evaluate alpha mat per element
                if self.mode == "per_token":
                    T = x.shape[1]
                    a_slice = self.alpha[:T].to(dtype).view(1, -1, 1)
                    t = torch.tanh(a_slice * x)
                    slopes = (a_slice * (1.0 - t * t)).abs()
                elif self.mode == "per_token_channel":
                    T = x.shape[1]
                    a_mat = self.alpha[:T, :x.shape[-1]].to(dtype).view(1, T, x.shape[-1])
                    t = torch.tanh(a_mat * x)
                    slopes = (a_mat * (1.0 - t * t)).abs()
                else:  # factorized
                    T = x.shape[1]
                    U = self.u[:T, :]
                    V = self.v
                    alpha_mat = (U @ V) * self.scalar_init
                    a = alpha_mat.to(dtype).view(1, T, x.shape[-1])
                    t = torch.tanh(a * x)
                    slopes = (a * (1.0 - t * t)).abs()
            else:
                raise RuntimeError("Unsupported mode in local_slope_stats")
            slopes_np = slopes.detach().cpu().numpy().ravel()
            return {
                "max_abs_slope": float(slopes_np.max()),
                "mean_abs_slope": float(slopes_np.mean()),
                "std_abs_slope": float(slopes_np.std()),
            }


# -------------------------
# Example quick integration snippet (commentary for user)
# -------------------------
# In your Transformer block file (where LayerNorm was used), replace:
#    self.norm1 = nn.LayerNorm(embed_dim)
# with:
#    from models.dynamic_tanh import DynamicTanh
#    self.norm1 = DynamicTanh(hidden_dim=embed_dim, mode="per_channel", init_alpha=1.0, use_affine=True)
#
# To instrument gradient flow in the training loop:
#    grad_container = {}
#    model.block3.norm1.attach_gradient_monitor("block3.norm1", grad_container)
#    ...
#    # after backward:
#    print(grad_container["block3.norm1"]["input_grad_norms"][-1], grad_container["block3.norm1"]["output_grad_norms"][-1])
#
# To snapshot alpha:
#    alpha_snapshot = model.block3.norm1.snapshot_alpha()
#    # store alpha_snapshot into logs
#
# To compute activation stats for tensor x (torch.Tensor):
#    stats = model.block3.norm1.compute_activation_stats(x)
