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

#include "src/fastertransformer/th_op/zcodepp/ZcodeppOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerZppEncoderModel::FasterTransformerZppEncoderModel(th::Tensor q_kernel,
                                                   th::Tensor q_bias,
                                                   th::Tensor k_kernel,
                                                   th::Tensor k_bias,
                                                   th::Tensor v_kernel,
                                                   th::Tensor v_bias,
                                                   th::Tensor attr_output_kernel,
                                                   th::Tensor attr_output_bias,
                                                   th::Tensor attr_output_layernorm_gamma,
                                                   th::Tensor attr_output_layernorm_beta,
                                                   th::Tensor inter_kernel,
                                                   th::Tensor inter_bias,
                                                   th::Tensor output_kernel,
                                                   th::Tensor output_bias,
                                                   th::Tensor output_layernorm_gamma,
                                                   th::Tensor output_layernorm_beta,
                                                   th::Tensor word_embedding_table,
                                                   th::Tensor word_embedding_layernorm_gamma,
                                                   th::Tensor word_embedding_layernorm_beta,
                                                   int64_t    head_num,
                                                   int64_t    head_size,
                                                   int64_t    inter_size,
                                                   int64_t    layer_num,
                                                   int64_t    position_buckets,
                                                   double     q_scaling):
    _st(at::ScalarType::Half),
    weights{q_kernel,
            q_bias,
            k_kernel,
            k_bias,
            v_kernel,
            v_bias,
            attr_output_kernel,
            attr_output_bias,
            attr_output_layernorm_gamma,
            attr_output_layernorm_beta,
            inter_kernel,
            inter_bias,
            output_kernel,
            output_bias,
            output_layernorm_gamma,
            output_layernorm_beta,
            word_embedding_table,
            word_embedding_layernorm_gamma,
            word_embedding_layernorm_beta}
{
    CHECK_INPUT(q_kernel, _st);                            // hidden_dim, hidden_dim
    CHECK_INPUT(q_bias, _st);                              // hidden_dim
    CHECK_INPUT(k_kernel, _st);                            // hidden_dim, hidden_dim
    CHECK_INPUT(k_bias, _st);                              // hidden_dim
    CHECK_INPUT(v_kernel, _st);                            // hidden_dim, hidden_dim
    CHECK_INPUT(v_bias, _st);                              // hidden_dim
    CHECK_INPUT(attr_output_kernel, _st);                  // hidden_dim, hidden_dim
    CHECK_INPUT(attr_output_bias, _st);                    // hidden_dim
    CHECK_INPUT(attr_output_layernorm_gamma, _st);         // hidden_dim
    CHECK_INPUT(attr_output_layernorm_beta, _st);          // hidden_dim
    CHECK_INPUT(inter_kernel, _st);                        // 4 * hidden_dim, hidden_dim
    CHECK_INPUT(inter_bias, _st);                          // 4 * hidden_dim
    CHECK_INPUT(output_kernel, _st);                       // hidden_dim, 4 * hidden_dim
    CHECK_INPUT(output_bias, _st);                         // hidden_dim
    CHECK_INPUT(output_layernorm_gamma, _st);              // hidden_dim
    CHECK_INPUT(output_layernorm_beta, _st);               // hidden_dim
    CHECK_INPUT(word_embedding_table, _st);                // vocab_size, hidden_dim
    CHECK_INPUT(word_embedding_layernorm_gamma, _st);      // hidden_dim
    CHECK_INPUT(word_embedding_layernorm_beta, _st);       // hidden_dim

    ftzppencoder = new FTZppEncoderModel<half>(head_num,
                                        head_size,
                                        inter_size,
                                        layer_num,
                                        position_buckets,
                                        q_scaling,
                                        weights);
    head_info    = torch::empty({5}, torch::dtype(torch::kInt64));
    head_info[0] = head_num;
    head_info[1] = head_size;
    head_info[2] = layer_num;
    head_info[3] = inter_size;
    head_info[4] = position_buckets;

    scaling_info    = torch::empty({1}, torch::dtype(torch::kFloat64));
    scaling_info[0] = (double)q_scaling;
}

FasterTransformerZppEncoderModel::~FasterTransformerZppEncoderModel()
{
    delete ftzppencoder;
}

th::Tensor FasterTransformerZppEncoderModel::forward(
    th::Tensor input, 
    th::Tensor sequence_lengths,
    std::vector<th::Tensor> pos_query_cache,
    std::vector<th::Tensor> pos_key_cache
)
{
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(sequence_lengths);
    CHECK_CONTIGUOUS(sequence_lengths);
    TORCH_CHECK(sequence_lengths.dtype() == torch::kInt32, "sequence_lengths dtype should be int32");
    long unsigned int batch_size = input.size(0);
    long unsigned int seq_len    = input.size(1);

    auto output =
        torch::empty({batch_size, seq_len, (head_info[0] * head_info[1]).item<int>()},
                     torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    ftzppencoder->forward(
        batch_size,
        seq_len,
        input, 
        sequence_lengths, 
        pos_query_cache,
        pos_key_cache,
        output
    );
    return output;
}

std::vector<th::Tensor> FasterTransformerZppEncoderModel::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    tmp.push_back(head_info);
    tmp.push_back(scaling_info);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerZppEncoderModelTHS =
torch::jit::class_<torch_ext::FasterTransformerZppEncoderModel>("FasterTransformer", "ZcodeppEncoder")
        .def(torch::jit::init<th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              th::Tensor,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              double>())
        .def("forward", &torch_ext::FasterTransformerZppEncoderModel::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::FasterTransformerZppEncoderModel>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::FasterTransformerZppEncoderModel> {
                int64_t head_num                  = state[19][0].item().to<int>();
                int64_t head_size                 = state[19][1].item().to<int>();
                int64_t layer_num                 = state[19][2].item().to<int>();
                int64_t inter_size                = state[19][3].item().to<int>();
                int64_t position_buckets          = state[19][4].item().to<int>();
                double  q_scaling                 = state[20][0].item().to<double>();
                return c10::make_intrusive<torch_ext::FasterTransformerZppEncoderModel>(state[0],
                                                                                state[1],
                                                                                state[2],
                                                                                state[3],
                                                                                state[4],
                                                                                state[5],
                                                                                state[6],
                                                                                state[7],
                                                                                state[8],
                                                                                state[9],
                                                                                state[10],
                                                                                state[11],
                                                                                state[12],
                                                                                state[13],
                                                                                state[14],
                                                                                state[15],
                                                                                state[16],
                                                                                state[17],
                                                                                state[18],
                                                                                head_num,
                                                                                head_size,
                                                                                inter_size,
                                                                                layer_num,
                                                                                position_buckets,
                                                                                q_scaling);
            });
