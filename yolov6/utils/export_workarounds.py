import torch
from torch import nn
import torch.nn.common_types as ct

FIX_TORCH_EXPORT_FULL = False
FIX_TF_LITE_2_11_MAXPOOL2D = False

def full(size, fill_value, dtype=None, device=None):
    if FIX_TORCH_EXPORT_FULL:
        return torch.ones(size, dtype=dtype, device=device) * fill_value
    return torch.full(size, fill_value, dtype=dtype, device=device)

def replace_maxpool2d(other: nn.MaxPool2d):
    if FIX_TF_LITE_2_11_MAXPOOL2D:
        return MaxPool2d.from_other(other)
    return other

class MaxPool2d(nn.MaxPool2d):
    """_summary_

    :param nn: _description_
    :return: _description_
    """
    @staticmethod
    def from_other(other: nn.MaxPool2d):
        return MaxPool2d(
            kernel_size=other.kernel_size,
            stride=other.stride,
            padding=other.padding,
            dilation=other.dilation,
            return_indices=other.return_indices,
            ceil_mode=other.ceil_mode,
        )


    def forward(self, input: torch.Tensor):
        pad = self.padding
        if isinstance(pad, int):
            pad = (pad + 1,) * 4
        else:
            pad = (pad[0] + 1, pad[0] + 1, pad[1] + 1, pad[1] + 1)
        x = nn.functional.pad(input, pad=pad, value=-float('inf'))
        x = x[..., 1:-1, 1:-1]
        x = nn.functional.max_pool2d(
          x, self.kernel_size, self.stride,
          padding=0, dilation=self.dilation, ceil_mode=self.ceil_mode,
          return_indices=self.return_indices)
        return x
