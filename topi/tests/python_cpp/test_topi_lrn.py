"""Test code for LRN"""
import numpy as np
import tvm
import topi
import logging
from topi.util import get_const_tuple
from itertools import product

def lrn_python(a_np, size, axis, bias, alpha, beta):
    """Local response normalization operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    size : int
        normalization window size

    axis : int
        input data layout channel axis

    bias : float
        offset to avoid dividing by 0. constant value

    alpha : float
        constant value

    beta : float
        exponent constant value

    Returns
    -------
    lrn_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    radius = size // 2
    sqr_sum = np.zeros(shape=a_np.shape).astype(a_np.dtype)
    for i, j, k, l in product(*[range(_axis) for _axis in a_np.shape]):
        axis_size = a_np.shape[axis]
        if (axis == 1):
            #NCHW layout
            sum_start = j-radius if j-radius >= 0 else 0
            sum_end = j+radius+1 if j+radius+1 < axis_size else axis_size
            sqr_sum[i, j, k, l] = sum(a_np[i, sum_start:sum_end, k, l] * \
                                      a_np[i, sum_start:sum_end, k, l])
        elif (axis == 3):
            #NHWC layout
            sum_start = l-radius if l-radius >= 0 else 0
            sum_end = l+radius+1 if l+radius+1 < axis_size else axis_size
            sqr_sum[i, j, k, l] = sum(a_np[i, j, k, sum_start:sum_end] * \
                                      a_np[i, j, k, sum_start:sum_end])

    sqr_sum_up = np.power((bias + (alpha * sqr_sum /size)), beta)
    lrn_out = np.divide(a_np, sqr_sum_up)
    return lrn_out

def verify_lrn(shape, size, axis, bias, alpha, beta):
    '''Verify Local response normalization operator by comparing outputs from tvm and numpy implementation'''
    A = tvm.placeholder(shape, name='A')
    B = topi.cpp.nn.lrn(A, size, axis, alpha, beta, bias)
    dtype = A.dtype

    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = lrn_python(a_np, size, axis, bias, alpha, beta)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        if device == "llvm":
            s = topi.cpp.generic.default_schedule(target, [B], False)
        else:
            s = topi.cpp.cuda.schedule_lrn(target, [B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-1)

    for device in ['cuda', 'opencl', 'metal', 'rocm', 'llvm']:
        check_device(device)

def test_lrn():
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 20, 20), 3, 1, 2.0, 1.0, 0.75)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_lrn()
