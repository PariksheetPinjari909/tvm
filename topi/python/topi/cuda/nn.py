from __future__ import absolute_import as _abs

import numpy as np
import tvm

from .. import generic
from .. import util
from .. import tag
from ..nn import pad
from ..nn.util import get_pad_tuple

@generic.schedule_lrn.register(["cuda"])
def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    num_thread = 64
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    lrn = outs[0]
    sqr_sum_up = lrn.op.input_tensors[1]
    sqr_sum = sqr_sum_up.op.input_tensors[0]
    pad = sqr_sum.op.input_tensors[0]
    s[pad].bind(pad.op.axis[0], block_x)
    rk = sqr_sum.op.reduce_axis[0]
    ko, ki = s[sqr_sum].split(rk, factor=num_thread)
    SF = s.rfactor(sqr_sum, ki)
    s[sqr_sum].bind(s[sqr_sum].op.axis[0], block_x)
    s[sqr_sum].bind(s[sqr_sum].op.reduce_axis[0], thread_x)
    s[SF].compute_at(s[sqr_sum], s[sqr_sum].op.reduce_axis[0])
    s[sqr_sum_up].bind(sqr_sum_up.op.axis[0], block_x)
    tx, xi = s[lrn].split(lrn.op.axis[1], nparts=num_thread)
    s[lrn].bind(lrn.op.axis[0], block_x)
    s[lrn].bind(tx, thread_x)
    return s
