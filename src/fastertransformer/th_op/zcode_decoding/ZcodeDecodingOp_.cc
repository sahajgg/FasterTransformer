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

#include "src/fastertransformer/th_op/zcode_decoding/ZcodeDecodingOp.h"

namespace th = torch;

namespace torch_ext {

template<typename T>
FTZcodeDecoding<T>::FTZcodeDecoding(int64_t                        head_num,
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
                              const std::vector<th::Tensor>& w):
    head_num_(head_num),
    size_per_head_(size_per_head),
    max_relative_positions_(max_relative_positions),
    relative_position_buckets_(relative_position_buckets),
    inter_size_(inter_size),
    d_model_(d_model),
    layer_num_(layer_num),
    vocab_size_(vocab_size),
    q_scaling_(q_scaling),
    end_id_(end_id),
    activation_type_(activation_type),
    _weights(w)
{
    ft::ftNcclInitialize(tensor_para_, pipeline_para_, 1, 1);

    ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
    cublas_algo_map_      = new ft::cublasAlgoMap("gemm_config.in");
    cublas_wrapper_mutex_ = new std::mutex();

    decoding_weights.decoder_layer_weights.clear();
    decoding_weights.decoder_layer_weights.resize(layer_num_);
    const int hidden_dim = head_num_ * size_per_head_;
    const int local_hidden_dim = (head_num_ / tensor_para_.world_size_) * size_per_head_;

    for (int i = 0; i < layer_num_; i++) {
        int local_num_layer = (int)(ceil(layer_num_ * 1.0f / pipeline_para_.world_size_));
        if (!(i < layer_num_ && (i >= local_num_layer * pipeline_para_.rank_)
                && (i < local_num_layer * (pipeline_para_.rank_ + 1)))) {
            continue;
        }
        const int first_layer_index = local_num_layer * pipeline_para_.rank_;

        decoding_weights.decoder_layer_weights[i].attention_weights.query_weight.kernel =
            get_ptr<T>(_weights[0]) + hidden_dim * local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.query_weight.bias =
            get_ptr<T>(_weights[1]) + local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.key_weight.kernel =
            get_ptr<T>(_weights[2]) + hidden_dim * local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.key_weight.bias =
            get_ptr<T>(_weights[3]) + local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.value_weight.kernel =
            get_ptr<T>(_weights[4]) + hidden_dim * local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.value_weight.bias =
            get_ptr<T>(_weights[5]) + local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.attention_output_weight.kernel =
            get_ptr<T>(_weights[6]) + local_hidden_dim * hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attention_weights.attention_output_weight.bias =
            get_ptr<T>(_weights[7]) + hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attn_layernorm_weights.gamma =
            get_ptr<T>(_weights[8]) + hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].attn_layernorm_weights.beta =
            get_ptr<T>(_weights[9]) + hidden_dim * (i - first_layer_index);

        decoding_weights.decoder_layer_weights[i].cross_attention_weights.query_weight.kernel =
            get_ptr<T>(_weights[10]) + hidden_dim * local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.query_weight.bias =
            get_ptr<T>(_weights[11]) + local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.key_weight.kernel =
            get_ptr<T>(_weights[12]) + hidden_dim * local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.key_weight.bias =
            get_ptr<T>(_weights[13]) + local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.value_weight.kernel =
            get_ptr<T>(_weights[14]) + hidden_dim * local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.value_weight.bias =
            get_ptr<T>(_weights[15]) + local_hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.attention_output_weight.kernel =
            get_ptr<T>(_weights[16]) + local_hidden_dim * hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attention_weights.attention_output_weight.bias =
            get_ptr<T>(_weights[17]) + hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attn_layernorm_weights.gamma =
            get_ptr<T>(_weights[18]) + hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].cross_attn_layernorm_weights.beta =
            get_ptr<T>(_weights[19]) + hidden_dim * (i - first_layer_index);
        
        decoding_weights.decoder_layer_weights[i].ffn_weights.intermediate_weight.kernel =
            get_ptr<T>(_weights[20])
            + hidden_dim * (inter_size_ / tensor_para_.world_size_) * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].ffn_weights.intermediate_weight.bias =
            get_ptr<T>(_weights[21]) + (inter_size_ / tensor_para_.world_size_) * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].ffn_weights.output_weight.kernel =
            get_ptr<T>(_weights[22])
            + (inter_size_ / tensor_para_.world_size_) * hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].ffn_weights.output_weight.bias =
            get_ptr<T>(_weights[23]) + hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].ffn_layernorm_weights.gamma =
            get_ptr<T>(_weights[24]) + hidden_dim * (i - first_layer_index);
        decoding_weights.decoder_layer_weights[i].ffn_layernorm_weights.beta =
            get_ptr<T>(_weights[25]) + hidden_dim * (i - first_layer_index);
    }

    // Model-level weights
    decoding_weights.word_embedding_table                       = get_ptr<T>(_weights[26]);
    decoding_weights.word_embedding_layernorm_weights.gamma     = get_ptr<T>(_weights[27]);
    decoding_weights.word_embedding_layernorm_weights.beta      = get_ptr<T>(_weights[28]);
    decoding_weights.lm_head_dense_weights.kernel               = get_ptr<T>(_weights[29]);
    decoding_weights.lm_head_dense_weights.bias                 = get_ptr<T>(_weights[30]);
    decoding_weights.lm_head_layernorm_weights.gamma            = get_ptr<T>(_weights[31]);
    decoding_weights.lm_head_layernorm_weights.beta             = get_ptr<T>(_weights[32]);
    decoding_weights.lm_head_bias                               = get_ptr<T>(_weights[33]);

    int device_id = 0;
    ft::check_cuda_error(cudaGetDevice(&device_id));
    ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
}

template<typename T>
std::vector<th::Tensor> FTZcodeDecoding<T>::forward(size_t                    max_seq_len,
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
                                                    th::optional<th::Tensor>& stop_words_list_opt)
{
    // input validation
    size_t beam_width = beam_width_opt.has_value() ? (size_t)beam_width_opt.value() : 1;
    uint   top_k      = top_k_opt.has_value() ? (uint)top_k_opt.value() : 1;
    float  top_p      = top_p_opt.has_value() ? (float)top_p_opt.value() : 0.0f;
    float  beam_search_diversity_rate =
        beam_search_diversity_rate_opt.has_value() ? (float)beam_search_diversity_rate_opt.value() : 0.0f;
    float temperature              = temperature_opt.has_value() ? (float)temperature_opt.value() : 1.0f;
    float len_penalty              = len_penalty_opt.has_value() ? (float)len_penalty_opt.value() : 0.0f;
    float repetition_penalty       = repetition_penalty_opt.has_value() ? (float)repetition_penalty_opt.value() : 1.0f;
    float presence_penalty         = presence_penalty_opt.has_value() ? (float)presence_penalty_opt.value() : 0.0f;
    int   min_length               = min_length_opt.has_value() ? (int)min_length_opt.value() : 0;
    unsigned long long random_seed = random_seed_opt.has_value() ? (unsigned long long)random_seed_opt.value() : 0;
    bool               is_return_output_log_probs =
        is_return_output_log_probs_opt.has_value() ? (bool)is_return_output_log_probs_opt.value() : false;
    bool is_return_cum_log_probs =
        is_return_cum_log_probs_opt.has_value() ? (bool)is_return_cum_log_probs_opt.value() : false;
    bool is_return_cross_attentions =
        is_return_cross_attentions_opt.has_value() ? (bool)is_return_cross_attentions_opt.value() : false;

    auto           stream       = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublasHandle, stream);
    ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
    ft::cublasMMWrapper*                  cublas_wrapper =
        ft::cublasMMWrapper(cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

    if (std::is_same<T, half>::value) {
        cublas_wrapper.setFP16GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        cublas_wrapper.setBF16GemmConfig();
    }
#endif
    else if (std::is_same<T, float>::value) {
        cublas_wrapper.setFP32GemmConfig();
    }

    const size_t batch_size      = (size_t)memory.size(0);
    const size_t seq_len         = (size_t)input.size(1);
    const size_t mem_max_seq_len = (size_t)memory.size(1);

    ft::DebertaDecoding<T> decoding =
        ft::DebertaDecoding<T>(batch_size,
                          max_seq_len,
                          mem_max_seq_len,
                          beam_width,
                          head_num_,
                          size_per_head_,
                          max_relative_positions_,
                          relative_position_buckets_,
                          inter_size_,
                          d_model_,
                          layer_num_,
                          vocab_size_,
                          q_scaling_,
                          -1,
                          end_id_,
                          beam_search_diversity_rate,
                          top_k,
                          top_p,
                          temperature,
                          len_penalty,
                          repetition_penalty,
                          stream,
                          cublas_wrapper,
                          allocator,
                          true,
                          &prop_,
                          activation_type_);
    ft::DataType data_type = ft::getTensorType<T>();

    ft::TensorMap input_tensors({
        {"input_ids", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, seq_len}, get_ptr<T>(input)}},
        {"encoder_output",
        ft::Tensor{ft::MEMORY_GPU, data_type,
                     std::vector<size_t>{(size_t)memory.size(0), (size_t)memory.size(1), (size_t)memory.size(2)},
                     get_ptr<T>(memory)}},
        {"encoder_sequence_length",
        ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32,
                     std::vector<size_t>{(size_t)memory_seq_lens.size(0)},
                     get_ptr<T>(memory_seq_lens)}},
        {"start_id", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{1}, get_ptr<T>(start_id)}}
    });

    ft::TensorMap pos_query_cache_map, pos_key_cache_map;
    for (uint l = 0; l < layer_num_; l++) {
        pos_query_cache_map.insert(
            "pos_query_cache_" + std::to_string(l), 
            ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, 2 * relative_position_buckets_, size_per_head_}, get_ptr<T>(pos_query_cache[l])}
        );
        pos_key_cache_map.insert(
            "pos_key_cache_" + std::to_string(l), 
            ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, head_num_, 2 * relative_position_buckets_, size_per_head_}, get_ptr<T>(pos_key_cache[l])}
        );
    }

    if (beam_width > 1) {
        input_tensors.insert(
            {"beam_search_diversity_rate",
             ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
    }
    if (top_p_opt.has_value()) {
        input_tensors.insert(
            {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
    }
    if (top_k_opt.has_value()) {
        input_tensors.insert(
            {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{1}, &top_k}});
    }
    if (temperature_opt.has_value()) {
        input_tensors.insert(
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}});
    }
    if (len_penalty_opt.has_value()) {
        input_tensors.insert(
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
    }
    if (repetition_penalty_opt.has_value()) {
        input_tensors.insert({"repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
    }
    if (presence_penalty_opt.has_value()) {
        input_tensors.insert(
            {"presence_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &presence_penalty}});
    }
    if (min_length_opt.has_value()) {
        input_tensors.insert(
            {"min_length", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &min_length}});
    }
    if (random_seed_opt.has_value()) {
        input_tensors.insert(
            {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});
    }
    if (stop_words_list_opt.has_value()) {
        input_tensors.insert({"stop_words_list", convert_tensor<int>(stop_words_list_opt.value())});
    }
    if (bad_words_list_opt.has_value()) {
        input_tensors.insert({"bad_words_list", convert_tensor<int>(bad_words_list_opt.value())});
    }

    auto output_ids      = torch::empty({(long int)(batch_size * beam_width * max_seq_len)},
                                   torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    auto sequence_length = torch::empty({(long int)(batch_size * beam_width)},
                                        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    std::vector<th::Tensor> th_output_tensors = {output_ids, sequence_length};

    ft::TensorMap output_tensors({{"output_ids",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_INT32,
                                              std::vector<size_t>{batch_size, beam_width, max_seq_len},
                                              get_ptr<int>(output_ids)}},
                                  {"sequence_length",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_INT32,
                                              std::vector<size_t>{batch_size, beam_width},
                                              get_ptr<int>(sequence_length)}}});

    try {
        decoding.forward(&output_tensors, &input_tensors, &pos_query_cache_map, &pos_key_cache_map, &decoding_weights);
    }
    catch (std::runtime_error& error) {
        std::cout << error.what();
        exit(-1);
    }
    catch (...) {
        std::cout << "Runtime error";
        exit(-1);
    }

    delete decoding;
    delete cublas_wrapper;
    delete allocator;
    return th_output_tensors;
}

template class FTZcodeDecoding<float>;
template class FTZcodeDecoding<half>;
#ifdef ENABLE_BF16
template class FTZcodeDecoding<__nv_bfloat16>;
#endif

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
                                int64_t              d_model,
                                int64_t              layer_num,
                                int64_t              vocab_size,
                                double               q_scaling,
                                int64_t              end_id,
                                std::string          activaiton_type):
    _st(q_kernel.scalar_type()),
    weights{
        q_kernel,
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
        lm_head_bias,
    }
{
    CHECK_INPUT(q_kernel, _st);                     // layer_num, d_model
    switch (_st) {
        case at::ScalarType::Float:
            ftdecoding = new torch_ext::FTZcodeDecoding<float>(head_num,
                                                            size_per_head,
                                                            max_relative_positions,
                                                            relative_position_buckets,
                                                            inter_size,
                                                            d_model,
                                                            layer_num,
                                                            vocab_size,
                                                            q_scaling,
                                                            end_id,
                                                            ft::getActivationType(activation_type),
                                                            weights);
            break;
        case at::ScalarType::Half:
            ftdecoding = new torch_ext::FTZcodeDecoding<half>(head_num,
                                                            size_per_head,
                                                            max_relative_positions,
                                                            relative_position_buckets,
                                                            inter_size,
                                                            d_model,
                                                            layer_num,
                                                            vocab_size,
                                                            q_scaling,
                                                            end_id,
                                                            ft::getActivationType(activation_type),
                                                            weights);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftdecoding = new torch_ext::FTZcodeDecoding<__nv_bfloat16>(head_num,
                                                            size_per_head,
                                                            max_relative_positions,
                                                            relative_position_buckets,
                                                            inter_size,
                                                            d_model,
                                                            layer_num,
                                                            vocab_size,
                                                            q_scaling,
                                                            end_id,
                                                            ft::getActivationType(activation_type),
                                                            weights);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

FasterTransformerZcodeDecoding::~FasterTransformerZcodeDecoding()
{
    delete ftdecoding;
}

std::vector<th::Tensor> FasterTransformerZcodeDecoding::forward(int64_t                   max_seq_len,
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
                                    th::optional<th::Tensor> stop_words_list)
{
    CHECK_INPUT(memory, _st);
    CHECK_TH_CUDA(memory_seq_lens);
    CHECK_CONTIGUOUS(memory_seq_lens);
    TORCH_CHECK(memory_seq_lens.dtype() == torch::kInt32, "mem_seq_lens dtype should be int32");

    auto results = ftdecoding->forward((size_t)max_seq_len,
                                       &input,
                                       &memory,
                                       &memory_seq_lens,
                                       &start_id,
                                       &pos_query_cache,
                                       &pos_key_cache,
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
                                       stop_words_list);
    return results;
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
                              int64_t,
                              double,
                              int64_t,
                              std::string>())
        .def("forward", &torch_ext::FasterTransformerZcodeDecoding::forward);
