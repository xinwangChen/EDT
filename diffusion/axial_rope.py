import math

import torch
import torch._dynamo
from torch import nn
from functools import lru_cache, reduce
from . import flags

if flags.get_use_compile():
    torch._dynamo.config.suppress_errors = True


def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    *shape, d, r = x.shape
    return x.view(*shape, d * r)


@flags.compile_wrap
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


def freqs_pixel(max_freq=10.0):
    def init(shape):
        freqs = torch.linspace(1.0, max_freq / 2, shape[-1]) * math.pi
        return freqs.log().expand(shape)
    return init


def freqs_pixel_log(max_freq=10.0):
    def init(shape):
        log_min = math.log(math.pi)
        log_max = math.log(max_freq * math.pi / 2)
        return torch.linspace(log_min, log_max, shape[-1]).expand(shape)
    return init

def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)

class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)