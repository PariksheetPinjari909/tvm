# pylint: disable=invalid-name
"""TVM operator for l2norm"""
from __future__ import absolute_import
import tvm
from .pad import pad

@tvm.target.generic_func
def l2norm_instance_nchw(data, eps):
    """Perform local response normalisation on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    eps : float
        epsilon value


    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    assert len(data.shape) == 4, "only support 4-dim lrn"
    b, c, h, w = data.shape
    ##Add padding on left & right of size radius first

    rh = tvm.reduce_axis((0 , h), name='rh')
    rw = tvm.reduce_axis((0 , w), name='rw')
    rc = tvm.reduce_axis((0 , c), name='rc')
    sqr_sum = tvm.compute((b), lambda i: tvm.sum(data[i, rc, rh, rw] * data[i, rc, rh, rw], axis=(rc, rh, rw)))

    sqrt_sum = tvm.compute((b), lambda i: tvm.intrin.sqrt(sqr_sum[i] + eps))

    return tvm.compute(
        data.shape, lambda b, c, h, w: data[b, c, h, w] / sqrt_sum[b])
