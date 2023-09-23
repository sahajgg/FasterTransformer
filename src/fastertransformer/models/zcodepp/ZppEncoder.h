/*
 * Copyright (c) 2023, sahajgg.  All rights reserved.
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

#include <vector>

#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/ZppEncoderAttentionLayer.h"
#include "src/fastertransformer/models/zcodepp/ZppEncoderWeight.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/kernels/WordEmbLookup.h"

namespace fastertransformer {

template<typename T>
class ZppEncoder: public BaseLayer {
private:
    // meta data
    size_t                      head_num_;
    size_t                      size_per_head_;
    size_t                      inter_size_;
    size_t                      hidden_units_;
    size_t                      num_layer_;
    size_t                      position_buckets_;
    static constexpr float      layernorm_eps_ = 1e-7f;
    float                       q_scaling_;

    BaseAttentionLayer<T>*      disentangled_attention_layer_ = nullptr;
    GeluFfnLayer<T>*            ffn_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer();
    void freeBuffer();
    void initialize();

    void allocateBuffer(size_t batch_size, size_t seq_len);

protected:
    // model params
    size_t* h_pinned_token_num_ptr_ = nullptr;
    int*    padding_offset_         = nullptr;
    T*      attention_mask_         = nullptr;
    T*      deberta_emb_buf_        = nullptr;
    T*      deberta_in_buffer_      = nullptr;
    T*      attn_out_buf_           = nullptr;
    T*      deberta_out_buffer_     = nullptr;

public:
    ZppEncoder(size_t                      head_num,
               size_t                      size_per_head,
               size_t                      inter_size,
               size_t                      num_layer,
               size_t                      position_buckets,
               float                       q_scaling,
               cudaStream_t                stream,
               cublasMMWrapper*            cublas_wrapper,
               IAllocator*                 allocator,
               bool                        is_free_buffer_after_forward);

    ~ZppEncoder();

    void forward(std::vector<Tensor>*          output_tensors,
                 const std::vector<Tensor>*    input_tensors,
                 const std::vector<Tensor>*    pos_query_cache,
                 const std::vector<Tensor>*    pos_key_cache,
                 const ZppEncoderWeight<T>*    zppencoder_weights);
    void forward(
        TensorMap* output_tensors, 
        TensorMap* input_tensors, 
        TensorMap* pos_query_cache,
        TensorMap* pos_key_cache,
        const ZppEncoderWeight<T>* zppencoder_weights);
};

}  // namespace fastertransformer
