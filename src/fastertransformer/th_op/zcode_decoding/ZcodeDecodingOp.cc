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

#include "src/fastertransformer/th_op/zcode_decoding/ZcodeDecodingOp.h"

namespace th = torch;
namespace torch_ext {

FasterTransformerZcodeDecoding::FasterTransformerZcodeDecoding(
                                th::Tensor           q_kernel,
                                th::Tensor           q_bias,
                                th::Tensor           k_kernel,
                                th::Tensor           k_bias,
                                th::Tensor           v_kernel,
                                th::Tensor           v_bias,
                                th::Tensor           attr_output_kernel,
                                th::Tensor           attr_output_bias,
                                th::Tensor           attr_output_layernorm_gamma,
                                th::Tensor           attr_output_layernorm_beta,
                                th::Tensor           cross_q_kernel,
                                th::Tensor           cross_q_bias,
                                th::Tensor           cross_k_kernel,
                                th::Tensor           cross_k_bias,
                                th::Tensor           cross_v_kernel,
                                th::Tensor           cross_v_bias,
                                th::Tensor           cross_attr_output_kernel,
                                th::Tensor           cross_attr_output_bias,
                                th::Tensor           cross_attr_output_layernorm_gamma,
                                th::Tensor           cross_attr_output_layernorm_beta,
                                th::Tensor           inter_kernel,
                                th::Tensor           inter_bias,
                                th::Tensor           output_kernel,
                                th::Tensor           output_bias,
                                th::Tensor           output_layernorm_gamma,
                                th::Tensor           output_layernorm_beta,
                                th::Tensor           word_embedding_table,
                                th::Tensor           word_embedding_layernorm_gamma,
                                th::Tensor           word_embedding_layernorm_beta,
                                th::Tensor           lm_head_dense_kernel,
                                th::Tensor           lm_head_dense_bias,
                                th::Tensor           lm_head_layernorm_gamma,
                                th::Tensor           lm_head_layernorm_beta,
                                th::Tensor           lm_head_bias,
                                int64_t              head_num,
                                int64_t              size_per_head,
                                int64_t              max_relative_positions,
                                int64_t              relative_position_buckets,
                                int64_t              inter_size,
                                int64_t              layer_num,
                                int64_t              vocab_size,
                                double               q_scaling,
                                int64_t              end_id):

    _st(q_kernel.scalar_type()),
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
            cross_q_kernel,
            cross_q_bias,
            cross_k_kernel,
            cross_k_bias,
            cross_v_kernel,
            cross_v_bias,
            cross_attr_output_kernel,
            cross_attr_output_bias,
            cross_attr_output_layernorm_gamma,
            cross_attr_output_layernorm_beta,
            inter_kernel,
            inter_bias,
            output_kernel,
            output_bias,
            output_layernorm_gamma,
            output_layernorm_beta,
            word_embedding_table,
            word_embedding_layernorm_gamma,
            word_embedding_layernorm_beta,
            lm_head_dense_kernel,
            lm_head_dense_bias,
            lm_head_layernorm_gamma,
            lm_head_layernorm_beta,
            lm_head_bias}
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

    CHECK_INPUT(cross_q_kernel, _st);                      // hidden_dim, hidden_dim
    CHECK_INPUT(cross_q_bias, _st);                        // hidden_dim
    CHECK_INPUT(cross_k_kernel, _st);                      // hidden_dim, hidden_dim
    CHECK_INPUT(cross_k_bias, _st);                        // hidden_dim
    CHECK_INPUT(cross_v_kernel, _st);                      // hidden_dim, hidden_dim
    CHECK_INPUT(cross_v_bias, _st);                        // hidden_dim
    CHECK_INPUT(cross_attr_output_kernel, _st);            // hidden_dim, hidden_dim
    CHECK_INPUT(cross_attr_output_bias, _st);              // hidden_dim
    CHECK_INPUT(cross_attr_output_layernorm_gamma, _st);   // hidden_dim
    CHECK_INPUT(cross_attr_output_layernorm_beta, _st);    // hidden_dim

    CHECK_INPUT(inter_kernel, _st);                        // 4 * hidden_dim, hidden_dim
    CHECK_INPUT(inter_bias, _st);                          // 4 * hidden_dim
    CHECK_INPUT(output_kernel, _st);                       // hidden_dim, 4 * hidden_dim
    CHECK_INPUT(output_bias, _st);                         // hidden_dim
    CHECK_INPUT(output_layernorm_gamma, _st);              // hidden_dim
    CHECK_INPUT(output_layernorm_beta, _st);               // hidden_dim
    CHECK_INPUT(word_embedding_table, _st);                // vocab_size, hidden_dim
    CHECK_INPUT(word_embedding_layernorm_gamma, _st);      // hidden_dim
    CHECK_INPUT(word_embedding_layernorm_beta, _st);       // hidden_dim

    switch (_st) {
        case at::ScalarType::Float:
            ftdeberta = new FTZcodeDecoding<float>(head_num,
                                             size_per_head,
                                             max_relative_positions,
                                             relative_position_buckets,
                                             inter_size,
                                             layer_num,
                                             vocab_size,
                                             q_scaling,
                                             end_id,
                                             weights);
            break;
        case at::ScalarType::Half:
            ftdeberta = new FTZcodeDecoding<half>(head_num,
                                             size_per_head,
                                             max_relative_positions,
                                             relative_position_buckets,
                                             inter_size,
                                             layer_num,
                                             vocab_size,
                                             q_scaling,
                                             end_id,
                                             weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftdeberta = new FTZcodeDecoding<__nv_bfloat16>(head_num,
                                             size_per_head,
                                             max_relative_positions,
                                             relative_position_buckets,
                                             inter_size,
                                             layer_num,
                                             vocab_size,
                                             q_scaling,
                                             end_id,
                                             weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

FasterTransformerZcodeDecoding::~FasterTransformerZcodeDecoding()
{
    delete ftdeberta;
}

std::vector<th::Tensor> FasterTransformerZcodeDecoding::forward(
    int64_t                  max_seq_len,
    th::Tensor               input,
    th::Tensor               memory,
    th::Tensor               memory_seq_lens,
    th::Tensor               start_id,
    std::vector<th::Tensor>  pos_query_cache,
    std::vector<th::Tensor>  pos_key_cache,
    th::optional<int64_t>    beam_width,
    th::optional<int64_t>    top_k,
    th::optional<double>     top_p,
    th::optional<double>     beam_search_diversity_rate,
    th::optional<double>     temperature,
    th::optional<double>     len_penalty,
    th::optional<double>     repetition_penalty,
    th::optional<double>     presence_penalty,
    th::optional<int64_t>    min_length,
    th::optional<int64_t>    random_seed,
    th::optional<bool>       is_return_output_log_probs,
    th::optional<bool>       is_return_cum_log_probs,
    th::optional<bool>       is_return_cross_attentions,
    th::optional<th::Tensor> bad_words_list,
    th::optional<th::Tensor> stop_words_list
)
{
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(memory_seq_lens);
    CHECK_CONTIGUOUS(memory_seq_lens);
    TORCH_CHECK(memory_seq_lens.dtype() == torch::kInt32, "sequence_lengths dtype should be int32");
    size_t batch_size = (size_t)input.size(0);
    size_t seq_len    = (size_t)input.size(1);

    auto output = ftdeberta->forward(
        max_seq_len,
        input,
        memory,
        memory_seq_lens,
        start_id,
        pos_query_cache,
        pos_key_cache,
        beam_width,
        top_k,
        top_p,
        beam_search_diversity_rate,
        temperature,
        len_penalty,
        repetition_penalty,
        presence_penalty,
        min_length,
        random_seed,
        is_return_output_log_probs,
        is_return_cum_log_probs,
        is_return_cross_attentions,
        bad_words_list,
        stop_words_list
    );
    return output;
}

std::vector<th::Tensor> FasterTransformerZcodeDecoding::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights);
    return tmp;
}

}  // namespace torch_ext

static auto FasterTransformerZcodeDecodingTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::FasterTransformerZcodeDecoding>("FasterTransformerZcodeDecoding")
#else
    torch::jit::class_<torch_ext::FasterTransformerZcodeDecoding>("FasterTransformer", "ZcodeDecoding")
#endif
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
                              int64_t,
                              int64_t,
                              double,
                              int64_t>())
        .def("forward", &torch_ext::FasterTransformerZcodeDecoding::forward);
