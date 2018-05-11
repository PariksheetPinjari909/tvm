/*!
*  Copyright (c) 2018 by Contributors
* \file rocm/nn.h
* \brief rocm schedule for lrn and l2 normalization operations
*/
#ifndef TOPI_ROCM_NN_H_
#define TOPI_ROCM_NN_H_

#include "tvm/tvm.h"
#include "tvm/build_module.h"
#include "topi/tags.h"

namespace topi {
using namespace tvm;
namespace rocm {
/*!
* \brief Create a rocm schedule for LRN
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_lrn(const Target &target, const Array<Tensor>& outs) {
  return topi::cuda::schedule_lrn(target, outs);
}

/*!
* \brief Create a rocm schedule for L2 Normalization
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_l2norm(const Target &target, const Array<Tensor>& outs) {
  return topi::cuda::schedule_l2norm(target, outs);
}
}  // namespace rocm
}  // namespace topi
#endif  // TOPI_ROCM_NN_H_
