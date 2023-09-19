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

#include "src/fastertransformer/layers/attention_layers/ZppEncoderAttentionLayer.h"
#include "src/fastertransformer/kernels/disentangled_attention_kernels.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"

namespace fastertransformer {

template<typename T>
void ZppEncoderAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                          TensorMap*                input_tensors,
                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, d_model],
    //      attention_mask [batch, 1, seqlen, seqlen],
    //      relative_embeddings [2*attention_span, hidden_size],
    //      padding_offset [token_num] (optional)
    //  output_tensors:
    //      hidden_features  [token_num, hidden_units]
    //      attentions [batch, num_layer, head_num, seqlen, seqlen] (optional)
    // If padding_offset.data is nullptr, then not remove padding

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t request_batch_size = input_tensors->at("attention_mask").shape[0];
    const size_t request_seq_len    = input_tensors->at("attention_mask").shape[2];
    allocateBuffer(request_batch_size, request_seq_len);

    T*         hidden_features     = output_tensors->getPtr<T>("hidden_features");
    const T*   from_tensor         = input_tensors->getPtr<T>("input_query");
    const T*   attention_mask      = input_tensors->getPtr<T>("attention_mask");
    const int* padding_offset      = input_tensors->getPtr<int>("padding_offset", nullptr);
    const T*   pos_query_cache     = input_tensors->getPtr<T>("pos_query_cache");
    const T*   pos_key_cache       = input_tensors->getPtr<T>("pos_key_cache");

    const int m = input_tensors->at("input_query").shape[0];  // total_valid_tokens
    int       k = d_model_;                                   // hidden size
    int       n = hidden_units_;                              // num_heads * head_size
    int       s = position_buckets_;                          // relative attention span ("k" in original paper)

    // Compute Q,K,V [token_num, hidden_size] --> [token_num, num_heads*head_size]
    const T* hA[]{attention_weights->query_weight.kernel,
                  attention_weights->key_weight.kernel,
                  attention_weights->value_weight.kernel,
                  nullptr,
                  from_tensor,
                  from_tensor,
                  from_tensor,
                  nullptr,
                  q_buf_,
                  k_buf_,
                  v_buf_,
                  nullptr};
    
    // Note: Here, we assume the weights of each time may be different.
    // If we can preprocess these weights before inference, we can reduce the overhead
    // caused by cudaMemcpyAsync
    cudaMemcpyAsync((void*)batch_qkv_kernel_ptr_, hA, sizeof(T*) * 12, cudaMemcpyHostToDevice, stream_);
    cublas_wrapper_->batchedGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    n,
                                    m,
                                    k,
                                    (const void* const*)batch_qkv_kernel_ptr_,
                                    n,
                                    (const void* const*)batch_qkv_input_ptr_,
                                    k,
                                    (void* const*)batch_qkv_buf_ptr_,
                                    n,
                                    3);

    // add QKV bias (bias optional, can be nullptr) & permute
    // [batch, seq_len, num_heads*head_size] or [token_num, num_heads*head_size] --> [batch, num_heads, seq_len,
    // head_size] Note: aligned to padded seq len again
    if (padding_offset == nullptr) {
        invokeAddQKVBiasIA3Transpose(q_buf_2_,
                                     k_buf_2_,
                                     v_buf_2_,
                                     q_buf_,
                                     attention_weights->query_weight.bias,
                                     k_buf_,
                                     attention_weights->key_weight.bias,
                                     v_buf_,
                                     attention_weights->value_weight.bias,
                                     request_batch_size,
                                     request_seq_len,
                                     head_num_,
                                     size_per_head_,
                                     (int*)nullptr,  // suppress IA3 inputs
                                     (T*)nullptr,
                                     (T*)nullptr,
                                     stream_);
        sync_check_cuda_error();
    }
    else {
        cudaMemsetAsync(q_buf_2_, 0, 3 * request_batch_size * request_seq_len * hidden_units_ * sizeof(T), stream_);
        sync_check_cuda_error();
        invokeAddQKVBiasIA3RebuildPadding(q_buf_,
                                          attention_weights->query_weight.bias,
                                          k_buf_,
                                          attention_weights->key_weight.bias,
                                          v_buf_,
                                          attention_weights->value_weight.bias,
                                          q_buf_2_,
                                          k_buf_2_,
                                          v_buf_2_,
                                          request_batch_size,
                                          request_seq_len,
                                          head_num_,
                                          size_per_head_,
                                          m,
                                          padding_offset,
                                          (int*)nullptr,  // suppress IA3 inputs
                                          (T*)nullptr,
                                          (T*)nullptr,
                                          stream_);
        sync_check_cuda_error();
    }

    float scalar = 1 / (sqrtf(size_per_head_ * 1.0f) * q_scaling_);

    // compute Q*K [batch, num_heads, q_seq_len, k_seq_len]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        request_seq_len,
                                        request_seq_len,
                                        size_per_head_,
                                        k_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        q_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        request_batch_size * head_num_, /* batch size */
                                        scalar /* alpha */);

    // above is content-to-content "c2c" attention, Qc*Kc^T
    // similarly, disentangled attention has two extra type of attentions (replacing the normal relative attention bias
    // w/ real attentions)

    // compute content-to-position "c2p" attention,  Qc*Kr^T [batch, num_heads, seq_len, 2*attention_span]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2 * s,
                                        request_seq_len,
                                        size_per_head_,
                                        pos_key_cache,
                                        size_per_head_,
                                        2 * s * size_per_head_,
                                        q_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        QcKr_buf_,
                                        2 * s,
                                        request_seq_len * 2 * s,
                                        request_batch_size * head_num_, /* batch size */
                                        scalar /* alpha */);
    
    // compute position-to-content "p2c" attention,  Kc*Qr^T [batch, num_heads, seq_len, 2*attention_span]
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        2 * s,
                                        request_seq_len,
                                        size_per_head_,
                                        pos_query_cache,
                                        size_per_head_,
                                        2 * s * size_per_head_,
                                        k_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        KcQr_buf_,
                                        2 * s,
                                        request_seq_len * 2 * s,
                                        request_batch_size * head_num_, /* batch size */
                                        scalar /* alpha */);

    // gather & add c2c+c2p+p2c. In-place operation
    invokeDisentangledAttention(
        qk_buf_, qk_buf_, QcKr_buf_, KcQr_buf_, request_batch_size * head_num_, request_seq_len, s, stream_);
    sync_check_cuda_error();

    // softmax(QK)
    MaskedSoftmaxParam<T, T> param;
    param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
    param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
    param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
    param.batch_size         = request_batch_size;
    param.q_length           = request_seq_len;
    param.k_length           = request_seq_len;
    param.num_heads          = head_num_;
    param.qk_scale           = 1.0f;
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream_);
    sync_check_cuda_error();

    // compute softmax(QK) * V
    cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        size_per_head_,
                                        request_seq_len,
                                        request_seq_len,
                                        v_buf_2_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        qk_buf_,
                                        request_seq_len,
                                        request_seq_len * request_seq_len,
                                        qkv_buf_,
                                        size_per_head_,
                                        request_seq_len * size_per_head_,
                                        request_batch_size * head_num_);

    // permute [batch_size, num_heads, seq_len, head_size] --> [batch_size, seq_len, num_heads, head_size] or
    // [token_num, num_heads, head_size] w/ padding removal
    if (padding_offset == nullptr) {
        invokeTransposeQKV(qkv_buf_2_,
                           qkv_buf_,
                           request_batch_size,
                           request_seq_len,
                           head_num_,
                           size_per_head_,
                           (float*)nullptr,
                           0,
                           stream_);
        sync_check_cuda_error();
    }
    else {
        invokeTransposeAttentionOutRemovePadding(qkv_buf_,
                                                 qkv_buf_2_,
                                                 m,
                                                 request_batch_size,
                                                 request_seq_len,
                                                 head_num_,
                                                 size_per_head_,
                                                 padding_offset,
                                                 (float*)nullptr,
                                                 0,
                                                 stream_);
    }

    // switch Linear dimension
    k = hidden_units_;
    n = d_model_;

    // attention output Linear layer (bias and layernorm are handled outside)
    cublas_wrapper_->Gemm(CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            attention_weights->attention_output_weight.kernel,
                            n,
                            qkv_buf_2_,
                            k,
                            hidden_features,
                            n);

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template<typename T>
ZppEncoderAttentionLayer<T>::ZppEncoderAttentionLayer(size_t           max_batch_size,
                                                          size_t           max_seq_len,
                                                          size_t           head_num,
                                                          size_t           size_per_head,
                                                          size_t           d_model,
                                                          size_t           position_buckets,
                                                          float            q_scaling,
                                                          cudaStream_t     stream,
                                                          cublasMMWrapper* cublas_wrapper,
                                                          IAllocator*      allocator,
                                                          bool             is_free_buffer_after_forward):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num_ * size_per_head_),
    d_model_(d_model),
    position_buckets_(position_buckets),
    q_scaling_(q_scaling)
{}

template<typename T>
ZppEncoderAttentionLayer<T>::~ZppEncoderAttentionLayer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void ZppEncoderAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ZppEncoderAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    q_buf_   = (T*)allocator_->reMalloc(q_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    k_buf_   = (T*)allocator_->reMalloc(k_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    v_buf_   = (T*)allocator_->reMalloc(v_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * 3 * batch_size * seq_len * hidden_units_, false);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * hidden_units_;
    qk_buf_  = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * head_num_ * seq_len * seq_len, false);

    QcKr_buf_    = (T*)allocator_->reMalloc(QcKr_buf_, sizeof(T) * 2 * batch_size * head_num_ * seq_len * 2 * position_buckets_, false);
    KcQr_buf_  = QcKr_buf_ + batch_size * head_num_ * seq_len * 2 * position_buckets_;
    qkv_buf_   = (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    batch_qkv_kernel_ptr_    = (T**)allocator_->reMalloc(batch_qkv_kernel_ptr_, sizeof(T*) * 12, false);
    batch_qkv_input_ptr_     = batch_qkv_kernel_ptr_ + 4;
    batch_qkv_buf_ptr_       = batch_qkv_input_ptr_ + 4;

    is_allocate_buffer_ = true;
}

template<typename T>
void ZppEncoderAttentionLayer<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&q_buf_));
        allocator_->free((void**)(&k_buf_));
        allocator_->free((void**)(&v_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&QcKr_buf_));
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&batch_qkv_kernel_ptr_));
        sync_check_cuda_error();
        is_allocate_buffer_ = false;
    }
}

template class ZppEncoderAttentionLayer<half>;

}  // namespace fastertransformer
