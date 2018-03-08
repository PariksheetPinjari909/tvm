# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Grouped Convolution in python"""
import numpy as np
import scipy.signal
import topi
from .conv2d_nchw_python import conv2d_nchw_python
from .conv2d_hwcn_python import conv2d_hwcn_python
from .conv2d_nhwc_python import conv2d_nhwc_python

def grouped_conv2d_nchw_python(a_np, w_np, groups, stride, padding):
    """Grouped Convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    groups : int
        Filter groups, this indicate the number of split convolution have to
        perform

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    a_nps = np.split(a_np, groups, 1)
    w_nps = np.split(w_np, groups, 0)

    conv_out = []
    for data, kernel in zip(a_nps, w_nps):
        conv_out.append(conv2d_nchw_python(data, kernel, stride, padding))

    return np.concatenate(conv_out, 1)

def grouped_conv2d_hwcn_python(a_np, w_np, groups, stride, padding):
    """Grouped Convolution operator in HWCN layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [in_height, in_width, in_channel, batch]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    groups : int
        Filter groups, this indicate the number of split convolution have to
        perform

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [out_height, out_width, out_channel, batch]
    """

    a_nps = np.split(a_np, groups, 2)
    w_nps = np.split(w_np, groups, 3)

    conv_out = []
    for data, kernel in zip(a_nps, w_nps):
        conv_out.append(conv2d_hwcn_python(data, kernel, stride, padding))

    return np.concatenate(conv_out, 2)

def grouped_conv2d_nhwc_python(a_np, w_np, groups, stride, padding):
    """Grouped Convolution operator in NHWC layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    groups : int
        Filter groups, this indicate the number of split convolution have to
        perform

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_height,  out_width, out_channel]
    """

    a_nps = np.split(a_np, groups, 3)
    w_nps = np.split(w_np, groups, 3)

    conv_out = []
    for data, kernel in zip(a_nps, w_nps):
        conv_out.append(conv2d_nhwc_python(data, kernel, stride, padding))

    return np.concatenate(conv_out, 3)
