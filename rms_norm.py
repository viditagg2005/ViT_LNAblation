import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, channels_last, eps=1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Compute the RMS over the last dimension (channel)
        if self.channels_last:
            rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            x = x / rms
            x = x * self.weight + self.bias
        else:
            # For (B, C, H, W) layout, normalize over channel dimension
            rms = x.pow(2).mean(1, keepdim=True).add(self.eps).sqrt()
            x = x / rms
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, channels_last={self.channels_last}"

def convert_ln_to_rms(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = RMSNorm(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_rms(child))
    del module
    return module_output
