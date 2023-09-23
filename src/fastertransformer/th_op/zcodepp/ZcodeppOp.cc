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

FasterTransformerZppEncoderModel::FasterTransformerZppEncoderModel(std::vector<th::Tensor> model_weights,
                                                   int64_t    head_num,
                                                   int64_t    head_size,
                                                   int64_t    inter_size,
                                                   int64_t    layer_num,
                                                   int64_t    position_buckets,
                                                   double     q_scaling):
    _st(at::ScalarType::Half),
    weights(model_weights)
{
    CHECK_INPUT(model_weights.at(3), _st);                            // hidden_dim, hidden_dim
    CHECK_INPUT(model_weights.at(4), _st);                              // hidden_dim
    CHECK_INPUT(model_weights.at(5), _st);                            // hidden_dim, hidden_dim
    CHECK_INPUT(model_weights.at(6), _st);                              // hidden_dim
    CHECK_INPUT(model_weights.at(7), _st);                            // hidden_dim, hidden_dim
    CHECK_INPUT(model_weights.at(8), _st);                              // hidden_dim
    CHECK_INPUT(model_weights.at(9), _st);                  // hidden_dim, hidden_dim
    CHECK_INPUT(model_weights.at(10), _st);                    // hidden_dim
    CHECK_INPUT(model_weights.at(11), _st);         // hidden_dim
    CHECK_INPUT(model_weights.at(12), _st);          // hidden_dim
    CHECK_INPUT(model_weights.at(13), _st);                        // 4 * hidden_dim, hidden_dim
    CHECK_INPUT(model_weights.at(14), _st);                          // 4 * hidden_dim
    CHECK_INPUT(model_weights.at(15), _st);                       // hidden_dim, 4 * hidden_dim
    CHECK_INPUT(model_weights.at(16), _st);                         // hidden_dim
    CHECK_INPUT(model_weights.at(17), _st);              // hidden_dim
    CHECK_INPUT(model_weights.at(18), _st);               // hidden_dim
    CHECK_INPUT(model_weights.at(0), _st);                // vocab_size, hidden_dim
    CHECK_INPUT(model_weights.at(1), _st);      // hidden_dim
    CHECK_INPUT(model_weights.at(2), _st);       // hidden_dim

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
    size_t batch_size = input.size(0);
    size_t seq_len    = input.size(1);

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
    tmp.insert(tmp.begin(), scaling_info);
    tmp.insert(tmp.begin(), head_info);
    return tmp;
}

}  // namespace torch_ext

static auto fasterTransformerZppEncoderModelTHS =
torch::jit::class_<torch_ext::FasterTransformerZppEncoderModel>("FasterTransformer", "ZcodeppEncoder")
        .def(torch::jit::init<std::vector<th::Tensor>,
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
                int64_t head_num                  = state[0][0].item().to<int>();
                int64_t head_size                 = state[0][1].item().to<int>();
                int64_t layer_num                 = state[0][2].item().to<int>();
                int64_t inter_size                = state[0][3].item().to<int>();
                int64_t position_buckets          = state[0][4].item().to<int>();
                double  q_scaling                 = state[1][0].item().to<double>();
                std::vector<th::Tensor> model_weights(state.begin() + 2, state.end());
                return c10::make_intrusive<torch_ext::FasterTransformerZppEncoderModel>(model_weights,
                                                                                head_num,
                                                                                head_size,
                                                                                inter_size,
                                                                                layer_num,
                                                                                position_buckets,
                                                                                q_scaling);
            });
