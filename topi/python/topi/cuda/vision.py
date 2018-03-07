# pylint: disable=invalid-name, unused-variable
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic

@generic.schedule_reorg.register(["cuda", "gpu"])
def schedule_reorg(outs):
    """Schedule for reorg operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of reorg
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for reorg.
    """
    target = tvm.target.current_target()
    if target.target_name == "cuda" and "cublas" in target.libs:
        return generic.schedule_extern(outs)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule_reorg(reorg_op):
        num_thread = 64#tvm.target.current_target(allow_none=False).max_num_threads
        bx, tx = s[reorg_op].split(reorg_op.op.axis[0], factor=num_thread)
        s[reorg_op].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[reorg_op].bind(tx, tvm.thread_axis((0, num_thread), "threadIdx.x"))
        output = outs[0].op.output(0)
        s[reorg_op].compute_at(s[output], s[output].op.axis[1])
        tx, xi = s[output].split(output.op.axis[0], nparts=num_thread)
        s[output].bind(tx, tvm.thread_axis((0, num_thread), "threadIdx.x"))
        s[reorg_op].set_store_predicate(tx.var.equal(0))
        s[output].set_store_predicate(tx.var.equal(0))

    def _traverse(op):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        # schedule reorg
        elif op.tag == 'reorg':
            reorg_op = op.output(0)
            _schedule_reorg(reorg_op)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)
    _traverse(outs[0].op)
    return s

@generic.schedule_shortcut.register(["cuda", "gpu"])
def schedule_shortcut(outs):
    """Schedule for shortcut operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of shortcut
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for shortcut.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule_shortcut(shortcut_op):
        num_thread = 64#tvm.target.current_target(allow_none=False).max_num_threads
        if shortcut_op.op in s.outputs:
            output = shortcut_op
        else:
            output = outs[0].op.output(0)
            s[shortcut_op].compute_at(s[output], s[output].op.axis[1])
        k = output.op.axis[0]
        bx, tx = s[output].split(k, factor=num_thread)
        s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[output].bind(tx, tvm.thread_axis("threadIdx.x"))

    def _traverse(op):
        # inline all one-to-one-mapping operators except the last stage (output)
        # schedule shortcut
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        elif op.tag == 'shortcut':
            shortcut_op = op.output(0)
            _schedule_shortcut(shortcut_op)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)
    _traverse(outs[0].op)
    return s

@generic.schedule_region.register(["cuda", "gpu"])
def schedule_region(outs):
    """Schedule for region operator.
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of region
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for region.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    output = outs[0].op.output(0)
    num_thread = 64#tvm.target.current_target(allow_none=False).max_num_threads
    def _schedule_softmax(softmax_op):
        softmax = softmax_op.input_tensors[0]
        max_elem = softmax_op.input_tensors[1]
        expsum = softmax_op.input_tensors[2]
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        s[max_elem].bind(max_elem.op.axis[0], block_x)
        k = expsum.op.reduce_axis[0]
        ko, ki = s[expsum].split(k, factor=num_thread)
        ef = s.rfactor(expsum, ki)
        s[expsum].bind(s[expsum].op.axis[0], block_x)
        s[expsum].bind(s[expsum].op.reduce_axis[0], thread_x)
        s[ef].compute_at(s[expsum], s[expsum].op.reduce_axis[0])
        s[expsum].set_store_predicate(thread_x.var.equal(0))
        tx, xi = s[softmax_op].split(softmax_op.axis[1], nparts=num_thread)
        s[softmax_op].bind(softmax_op.axis[0], block_x)
        s[softmax_op].bind(tx, thread_x)
        return max_elem.op.input_tensors[0]

    def _traverse(op):
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        elif op.tag == 'softmax_output':
            tensor = _schedule_softmax(op)
            if tensor.op.input_tensors:
                _traverse(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)
    _traverse(outs[0].op)
    k = output.op.axis[0]
    bx, tx = s[output].split(k, factor=num_thread)
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s
