import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

class DynamicArctan(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.beta = nn.Parameter(torch.ones(1) * beta_init)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        y = torch.atan(self.alpha * x) * self.beta
        if self.channels_last:
            y = y * self.weight + self.bias
        else:
            y = y * self.weight[:, None, None] + self.bias[:, None, None]
        return y

    def extra_repr(self):
        return f"shape={self.normalized_shape}, α={float(self.alpha)}, β={float(self.beta)}"

def convert_ln_to_dyatan(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicArctan(
            module.normalized_shape,
            not isinstance(module, LayerNorm2d)
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyatan(child))
    del module
    return module_output
