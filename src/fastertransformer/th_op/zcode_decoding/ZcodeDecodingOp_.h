/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/models/zcode_decoding/DebertaDecoding.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class IFTZcodeDecoding {
public:
    virtual ~IFTZcodeDecoding()                                       = default;
    virtual std::vector<th::Tensor> forward(size_t                    max_seq_len,
                                            th::Tensor&               input,
                                            th::Tensor&               memory,
                                            th::Tensor&               memory_seq_lens,
                                            th::Tensor&               start_id,
                                            std::vector<th::Tensor>&  pos_query_cache,
                                            std::vector<th::Tensor>&  pos_key_cache,
                                            th::optional<int64_t>     beam_width_opt,
                                            th::optional<int64_t>     top_k_opt,
                                            th::optional<double>      top_p_opt,
                                            th::optional<double>      beam_search_diversity_rate_opt,
                                            th::optional<double>      temperature_opt,
                                            th::optional<double>      len_penalty_opt,
                                            th::optional<double>      repetition_penalty_opt,
                                            th::optional<double>      presence_penalty_opt,
                                            th::optional<int64_t>     min_length_opt,
                                            th::optional<int64_t>     random_seed_opt,
                                            th::optional<bool>        is_return_output_log_probs_opt,
                                            th::optional<bool>        is_return_cum_log_probs_opt,
                                            th::optional<bool>        is_return_cross_attentions_opt,
                                            th::optional<th::Tensor>& bad_words_list,
                                            th::optional<th::Tensor>& stop_words_list) = 0;
};

template<typename T>
class FTZcodeDecoding: public IFTZcodeDecoding {
public:
    FTZcodeDecoding(int64_t                        head_num,
                 int64_t                        size_per_head,
                 int64_t                        max_relative_positions,
                 int64_t                        relative_position_buckets,
                 int64_t                        inter_size,
                 int64_t                        d_model,
                 int64_t                        layer_num,
                 int64_t                        vocab_size,
                 double                         q_scaling,
                 int64_t                        end_id,
                 ft::ActivationType             activation_type,
                 const std::vector<th::Tensor>& w);

    ~FTZcodeDecoding() override
    {
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    std::vector<th::Tensor> forward(size_t                    max_seq_len,
                                    th::Tensor&               input,
                                    th::Tensor&               memory,
                                    th::Tensor&               memory_seq_lens,
                                    th::Tensor&               start_id,
                                    std::vector<th::Tensor>&  pos_query_cache,
                                    std::vector<th::Tensor>&  pos_key_cache,
                                    th::optional<int64_t>     beam_width_opt,
                                    th::optional<int64_t>     top_k_opt,
                                    th::optional<double>      top_p_opt,
                                    th::optional<double>      beam_search_diversity_rate_opt,
                                    th::optional<double>      temperature_opt,
                                    th::optional<double>      len_penalty_opt,
                                    th::optional<double>      repetition_penalty_opt,
                                    th::optional<double>      presence_penalty_opt,
                                    th::optional<int64_t>     min_length_opt,
                                    th::optional<int64_t>     random_seed_opt,
                                    th::optional<bool>        is_return_output_log_probs_opt,
                                    th::optional<bool>        is_return_cum_log_probs_opt,
                                    th::optional<bool>        is_return_cross_attentions_opt,
                                    th::optional<th::Tensor>& bad_words_list_opt,
                                    th::optional<th::Tensor>& stop_words_list_opt) override;

private:
    const int64_t                   head_num_;
    const int64_t                   size_per_head_;
    const int64_t                   max_relative_positions_;
    const int64_t                   relative_position_buckets_;
    const int64_t                   inter_size_;
    const int64_t                   d_model_;
    const int64_t                   layer_num_;
    const int64_t                   vocab_size_;
    double                          q_scaling_;
    const int64_t                   end_id_;
    const ft::ActivationType        activation_type_;

    std::vector<th::Tensor> _weights;
    cublasLtHandle_t        cublasltHandle_;
    std::mutex*             cublas_wrapper_mutex_;
    ft::cublasAlgoMap*      cublas_algo_map_;
    struct cudaDeviceProp   prop_;
    ft::ZcodeDecodingWeight<T> decoding_weights;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
};

class FasterTransformerZcodeDecoding: public torch::jit::CustomClassHolder {
public:
    FasterTransformerZcodeDecoding(
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
                                int64_t              d_model,
                                int64_t              layer_num,
                                int64_t              vocab_size,
                                double               q_scaling,
                                int64_t              end_id,
                                std::string          activaiton_type
    );

    ~FasterTransformerZcodeDecoding();

    std::vector<th::Tensor> forward(int64_t                   max_seq_len,
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
                                    th::optional<th::Tensor> stop_words_list);

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType      _st;
    torch_ext::IFTZcodeDecoding* ftdecoding;
    std::vector<th::Tensor>   weights;
};

}  // namespace torch_ext
