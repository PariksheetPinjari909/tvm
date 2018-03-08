
from __future__ import absolute_import as _abs

import numpy as np
import tvm
import topi
from .. import generic
from .. import util
from .. import tag
from ..nn import pad
from ..nn.util import get_pad_tuple

@generic.schedule_lrn.register(["rocm", "gpu"])
def schedule_lrn(outs):
    return topi.cuda.schedule_lrn(outs)

@generic.schedule_l2norm.register(["rocm", "gpu"])
def schedule_l2norm(outs):
    return topi.cuda.schedule_l2norm(outs)
