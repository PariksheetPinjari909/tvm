"""Shortcut operators (short-cut connections)."""
from __future__ import absolute_import as _abs
import tvm
import topi

def _simplify(shape):
    return int(str(shape[0])), int(str(shape[1])), int(str(shape[2])), int(str(shape[3]))

@tvm.target.generic_func
def shortcut(inp1, inp2):
    """Shortcut forward operators.

    Parameters
    ----------
    First Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Second Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    _, inp1_c, inp1_h, inp1_w = _simplify(inp1.shape)
    batch, inp2_c, inp2_h, inp2_w = _simplify(inp2.shape)

    stride = int(max(inp2_w / inp1_w, 1))
    sample = int(max(inp1_w / inp2_w, 1))
    minc = min(inp2_c, inp1_c)
    minh = min(inp2_h, inp1_h)
    minw = min(inp2_w, inp1_w)

    out = tvm.compute((batch, minc, minh, minw), lambda b, c, h, w:
                      inp1[b, c, h * sample, w * sample] +
                      inp2[b, c, h * stride, w * stride],
                      tag="shortcut")

    split_indices = int(inp1_c / minc)
    if split_indices > 1:
        split_res = topi.split(inp1, split_indices, 1)
        split_res[0] = out
        out = topi.concatenate(split_res, 1)

    return out
