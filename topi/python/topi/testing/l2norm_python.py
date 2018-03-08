"""L2 norm in python"""
import numpy as np
import mxnet as mx

def l2norm_nchw_python(a_np, eps):
    """L2 norm operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    eps : float
        epsilon constant value

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return mx.ndarray.L2Normalization(mx.nd.array(a_np), eps)

