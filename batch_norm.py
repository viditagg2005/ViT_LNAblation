import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class DynamicBatchNorm(nn.Module):
    def __init__(self, normalized_shape, channels_last, eps=1e-5, momentum=0.1):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.eps = eps
        self.momentum = momentum

        # Choose the right BatchNorm based on data layout
        if self.channels_last:
            # For channels_last, use BatchNorm1d (for [B, N, C] or [B, C])
            self.bn = nn.BatchNorm1d(normalized_shape, eps=eps, momentum=momentum)
        else:
            # For [B, C, H, W], use BatchNorm2d
            self.bn = nn.BatchNorm2d(normalized_shape, eps=eps, momentum=momentum)

    def forward(self, x):
        # If channels_last, permute to (B, C, *) for BatchNorm1d
        if self.channels_last:
            if x.dim() == 2:
                # [B, C] input
                return self.bn(x)
            elif x.dim() == 3:
                # [B, N, C] input: move C to second dim for BatchNorm1d
                x = x.permute(0, 2, 1)
                x = self.bn(x)
                x = x.permute(0, 2, 1)
                return x
            else:
                raise ValueError("Unsupported input shape for channels_last BatchNorm")
        else:
            # [B, C, H, W] input
            return self.bn(x)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, momentum={self.momentum}, channels_last={self.channels_last}"

def convert_ln_to_dynbn(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicBatchNorm(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dynbn(child))
    del module
    return module_output
