/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cstddef>
#include <vector>

#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/zcode_decoder/Deberta.h"
#include "src/fastertransformer/models/zcode_decoding/DebertaWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/kernels/activation_kernels.h"

namespace fastertransformer {

// fallback to fp32 dynamic decoder when bf16 specified
template<typename T>
struct fallBackType {
    using Type = float;
};

template<>
struct fallBackType<half> {
    using Type = half;
};

template<typename T>
class DebertaDecoding: public BaseLayer {
private:
    // meta data
    const size_t         head_num_;
    const size_t         size_per_head_;
    const size_t         inter_size_;
    const size_t         d_model_;
    const size_t         num_layer_;
    const size_t         max_relative_positions_;     // usually 2 * attention_span
    const size_t         relative_position_buckets_;  // i.e., attention_span
    const size_t         vocab_size_;
    const ActivationType activation_type_;
    float                q_scaling_;

    const int start_id_;
    const int end_id_;

    constexpr static float layernorm_eps_ = 1e-7f;

    // TODO(bhsueh) remove
    const float  beam_search_diversity_rate_;
    const size_t hidden_units_;
    const size_t top_k_;
    const float  top_p_;
    const float  temperature_;
    const float  len_penalty_;
    const float  repetition_penalty_;

    // calculated data
    size_t vocab_size_padded_;

    ZcodeDecoder<T>* decoder_;
    using DynamicDecodeType = typename fallBackType<T>::Type;
    DynamicDecodeLayer<DynamicDecodeType>* dynamic_decode_layer_;

    void allocateBuffer() override;
    void freeBuffer() override;
    void allocateBuffer(
        size_t batch_size, size_t beam_width, size_t decoder_seq_len, size_t max_seq_len, size_t max_mem_seq_len, size_t encoder_d_model);

    void initialize();

protected:
    T*       padded_embedding_kernel_                = nullptr;
    const T* padded_embedding_kernel_ptr_            = nullptr;

    T*              deberta_emb_buf_           = nullptr;
    int*      decoding_sequence_lengths_ = nullptr;

    T*                 decoder_input_buf_         = nullptr;
    T*                 decoder_output_buf_        = nullptr;
    
    T* logits_buf_iter                            = nullptr;
    T* logits_buf_                                = nullptr;
    float*             cum_log_probs_             = nullptr;
    bool*              finished_buf_              = nullptr;
    bool*              h_finished_buf_            = nullptr;

    int* start_ids_buf_ = nullptr;
    int* end_ids_buf_   = nullptr;

    T*   key_cache_             = nullptr;
    T*   value_cache_           = nullptr;
    T*   key_mem_cache_         = nullptr;
    T*   value_mem_cache_       = nullptr;
    T*   attention_mask_        = nullptr;
    int* cache_indirections_[2] = {nullptr, nullptr};

    int*   output_ids_buf_           = nullptr;
    int*   parent_ids_buf_           = nullptr;
    int*   output_ids_transpose_buf_ = nullptr;
    float* output_log_probs_buf_     = nullptr;

    T*   tiled_encoder_output_          = nullptr;
    int* tiled_encoder_sequence_length_ = nullptr;

    const T*   encoder_output_ptr_          = nullptr;
    const int* encoder_sequence_length_ptr_ = nullptr;

    const bool     using_beam_hyps = true;
    BeamHypotheses beam_hyps_;

    // function pointer callback
    using callback_sig                 = void(TensorMap*, void*);
    callback_sig* token_generated_cb_  = nullptr;
    void*         token_generated_ctx_ = nullptr;

public:
    DebertaDecoding(size_t                              max_batch_size,
                    size_t                              max_seq_len,
                    size_t                              mem_max_seq_len,
                    size_t                              beam_width,
                    size_t                              head_num,
                    size_t                              size_per_head,
                    size_t                              max_relative_positions,
                    size_t                              relative_position_buckets,
                    size_t                              inter_size,
                    size_t                              d_model,
                    size_t                              num_layer,
                    size_t                              vocab_size,
                    float                               q_scaling,
                    int                                 start_id,
                    int                                 end_id,
                    float                               beam_search_diversity_rate,
                    size_t                              top_k,
                    float                               top_p,
                    float                               temperature,
                    float                               len_penalty,
                    float                               repetition_penalty,
                    cudaStream_t                        stream,
                    cublasMMWrapper*                    cublas_wrapper,
                    IAllocator*                         allocator,
                    bool                                is_free_buffer_after_forward,
                    cudaDeviceProp*                     cuda_device_prop,
                    ActivationType                      activation_type = ActivationType::Gelu);

    DebertaDecoding(DebertaDecoding<T> const& DebertaDecoding);

    ~DebertaDecoding();

    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const TensorMap*           pos_query_cache,
                 const TensorMap*           pos_key_cache,
                 const ZcodeDecodingWeight<T>* decoding_weights);

    void forward(TensorMap*                      output_tensors, 
                 TensorMap*                      input_tensors, 
                 const TensorMap*                pos_query_cache,
                 const TensorMap*                pos_key_cache,
                 const ZcodeDecodingWeight<T>* decoding_weights);

    void registerCallback(callback_sig* fn, void* ctx);
    void unRegisterCallback();

    void setOutputTensors(TensorMap* output_tensors, const TensorMap* input_tensors);
};

}  // namespace fastertransformer
