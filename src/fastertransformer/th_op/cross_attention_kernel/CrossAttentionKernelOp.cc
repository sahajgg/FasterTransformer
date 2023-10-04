/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/th_op/cross_attention_kernel/CrossAttentionKernelOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerCrossAttentionKernel::FasterTransformerCrossAttentionKernel(
                                th::Tensor           cross_q_kernel,
                                th::Tensor           cross_q_bias,
                                th::Tensor           cross_k_bias,
                                th::Tensor           cross_v_bias,
                                th::Tensor           cross_attn_output_kernel,
                                th::Tensor           cross_attn_output_bias,
                                th::Tensor           cross_attn_layernorm_gamma,
                                th::Tensor           cross_attn_layernorm_beta,
                                int64_t              head_num,
                                int64_t              size_per_head,
                                double               q_scaling):

    _st(cross_q_kernel.scalar_type()),
    weights{cross_q_kernel,
            cross_q_bias,
            cross_k_bias,
            cross_v_bias,
            cross_attn_output_kernel,
            cross_attn_output_bias,
            cross_attn_layernorm_gamma,
            cross_attn_layernorm_beta}
{
    ftcrossattentionkernel = new FTCrossAttentionKernel<half>(head_num,
                                        size_per_head,
                                        q_scaling,
                                        weights);
}

FasterTransformerCrossAttentionKernel::~FasterTransformerCrossAttentionKernel()
{
    delete ftcrossattentionkernel;
}

th::Tensor FasterTransformerCrossAttentionKernel::forward(
    th::Tensor               query,
    th::Tensor               key_cache,
    th::Tensor               value_cache,
    th::Tensor               encoder_sequence_length,
    th::Tensor               step,
    th::Tensor               finished
)
{
    size_t batch_size = (size_t)query.size(0);
    auto output = ftcrossattentionkernel->forward(
        query,
        key_cache,
        value_cache,
        encoder_sequence_length,
        step,
        finished
    );
    return output;
}

std::vector<th::Tensor> FasterTransformerCrossAttentionKernel::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    return tmp;
}

}  // namespace torch_ext

static auto FasterTransformerCrossAttentionKernelTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerCrossAttentionKernel>("FasterTransformerCrossAttentionKernel")
#else
    torch::jit::class_<torch_ext::FasterTransformerCrossAttentionKernel>("FasterTransformer", "CrossAttentionKernel")
#endif
        .def(torch::jit::init<th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              int64_t,
                              int64_t,
                              double>())
        .def("forward", &torch_ext::FasterTransformerCrossAttentionKernel::forward);
