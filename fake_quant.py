import torch
from torch import nn
from functools import partial

def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (in_channels, out_channels, kernel_size, kernel_size)
    w_shape = w.shape[0]
    w_reshape = w.reshape(w_shape, -1)
    w_max = w_reshape.abs().max(dim=-1)[0]
    w_max = w_max.reshape(w_shape, 1, 1, 1)
    q_max = 2**(n_bits-1)-1
    scales = w_max.clamp(min=1e-5).div(q_max)
    w = w.div(scales).round().mul(scales)
    return w

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    #t : (1, out_channels, H, W)
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales = scales.clamp(min=1e-5).div(q_max)
    t = t.div(scales).round().mul(scales)
    return t

class W8A8Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0]
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = int((kernel_size[0] - 1) / 2) * dilation
            

        self.register_buffer('weight', torch.randn(self.in_channels, self.out_channels,
                                                   self.kernel_size, self.kernel_size,
                                                   dtype=torch.float16, requires_grad=False))
        
        self.register_buffer('bias', torch.zeros(self.out_channels, dtype=torch.float16,
                                                 requires_grad=False))
        

        self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        
    def to(self, *args, **kwargs):
        super(W8A8Conv2D, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding,
                                      self.dilation, self.groups)        
        return y
    

    @staticmethod
    def from_float(module):
        assert isinstance(module, torch.nn.Conv2d)
        new_module = W8A8Conv2D(module.in_channels, module.out_channels,
                                module.kernel_size)
        new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=8)
        new_module.bias = module.bias

        return new_module
    
    def __repr__(self):
        return f'W8A8Conv2D({self.in_channels}, {self.out_channels}, bias={self.bias is not None})'