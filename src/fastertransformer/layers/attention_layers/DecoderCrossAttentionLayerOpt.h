/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/attention_layers/BaseAttentionLayer.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

template<typename T>
class DecoderCrossAttentionLayerOpt: public BaseAttentionLayer<T> {
private:
    // metadata
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t d_model_;
    bool         is_batch_major_cache_ = true;

    // calculated params
    const size_t hidden_units_;
    const float  q_scaling_;

    // buffer handling
    size_t max_batch_size_  = 0;
    size_t max_mem_seq_len_ = 0;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t max_mem_seq_len);
    void freeBuffer() override;

protected:
    using BaseAttentionLayer<T>::stream_;
    using BaseAttentionLayer<T>::is_free_buffer_after_forward_;
    using BaseAttentionLayer<T>::is_allocate_buffer_;
    using BaseAttentionLayer<T>::cublas_wrapper_;
    using BaseAttentionLayer<T>::allocator_;

    T* q_buf_         = nullptr;
    T* context_buf_   = nullptr;
    T* key_mem_cache = nullptr;
    T* value_mem_cache = nullptr;

public:
    DecoderCrossAttentionLayerOpt(size_t           max_batch_size,
                               size_t           head_num,
                               size_t           size_per_head,
                               cudaStream_t     stream,
                               cublasMMWrapper* cublas_wrapper,
                               IAllocator*      allocator,
                               bool             is_free_buffer_after_forward);

    DecoderCrossAttentionLayerOpt(size_t           max_batch_size,
                               size_t           head_num,
                               size_t           size_per_head,
                               const float      q_scaling,
                               cudaStream_t     stream,
                               cublasMMWrapper* cublas_wrapper,
                               IAllocator*      allocator,
                               bool             is_free_buffer_after_forward);

    DecoderCrossAttentionLayerOpt(size_t           max_batch_size,
                               size_t           head_num,
                               size_t           size_per_head,
                               size_t           d_model,
                               const float      q_scaling,
                               cudaStream_t     stream,
                               cublasMMWrapper* cublas_wrapper,
                               IAllocator*      allocator,
                               bool             is_free_buffer_after_forward);

    DecoderCrossAttentionLayerOpt(DecoderCrossAttentionLayerOpt<T> const& attention_layer);

    ~DecoderCrossAttentionLayerOpt();

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights) override;

    void
    forward(TensorMap* output_tensors, TensorMap* input_tensors, const AttentionWeight<T>* attention_weights, const LayerNormWeight<T>* layernorm_weights);
};

}  // namespace fastertransformer
