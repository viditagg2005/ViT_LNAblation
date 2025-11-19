import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class DynamicGeluClip(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init=1.0, clip_val_init=2.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.clip_val = nn.Parameter(torch.ones(1) * clip_val_init)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        y = torch.nn.functional.gelu(self.alpha * x)
        c = self.clip_val.abs()
        y = torch.clamp(y, -c, c)
        if self.channels_last:
            y = y * self.weight + self.bias
        else:
            y = y * self.weight[:, None, None] + self.bias[:, None, None]
        return y

    def extra_repr(self):
        return f"shape={self.normalized_shape}, α={float(self.alpha)}, clip={float(self.clip_val)}"

def convert_ln_to_dygeluclip(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicGeluClip(
            module.normalized_shape,
            not isinstance(module, LayerNorm2d)
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dygeluclip(child))
    del module
    return module_output
