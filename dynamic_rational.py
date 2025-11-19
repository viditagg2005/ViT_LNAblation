
import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

# 4. Parametric Rational: rational function (p(x) / q(x)) e.g. (a x + b x^3) / (1 + c x^2 + d x^4)

class DynamicRational(nn.Module):
    def __init__(self, normalized_shape, channels_last, a_init=1.0, b_init=0.0, c_init=0.0, d_init=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        # parameters for numerator and denominator
        self.a = nn.Parameter(torch.ones(1) * a_init)
        self.b = nn.Parameter(torch.ones(1) * b_init)
        self.c = nn.Parameter(torch.ones(1) * c_init)
        self.d = nn.Parameter(torch.ones(1) * d_init)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        num = self.a * x + self.b * x**3
        denom = 1.0 + self.c * x**2 + self.d * x**4
        y = num / denom
        if self.channels_last:
            y = y * self.weight + self.bias
        else:
            y = y * self.weight[:, None, None] + self.bias[:, None, None]
        return y

    def extra_repr(self):
        return (f"shape={self.normalized_shape}, a={float(self.a)}, b={float(self.b)}, "
                f"c={float(self.c)}, d={float(self.d)}")

def convert_ln_to_dyrational(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicRational(
            module.normalized_shape,
            not isinstance(module, LayerNorm2d)
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyrational(child))
    del module
    return module_output