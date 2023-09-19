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

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/models/zcodepp/ZppEncoder.h"

namespace fastertransformer {

template<typename T>
void ZppEncoder<T>::initialize()
{
    disentangled_attention_layer_ = new ZppEncoderAttentionLayer<T>(0,
                                                                    0,
                                                                    head_num_,
                                                                    size_per_head_,
                                                                    head_num_ * size_per_head_,
                                                                    position_buckets_,
                                                                    q_scaling_,
                                                                    stream_,
                                                                    cublas_wrapper_,
                                                                    allocator_,
                                                                    is_free_buffer_after_forward_);

    ffn_layer_ = new GeluFfnLayer<T>(0,
                                    0,
                                    head_num_,
                                    size_per_head_,
                                    0,
                                    inter_size_,
                                    stream_,
                                    cublas_wrapper_,
                                    allocator_,
                                    is_free_buffer_after_forward_,
                                    false,
                                    0,
                                    false);
}

template<typename T>
ZppEncoder<T>::ZppEncoder(size_t                              max_batch_size,
                          size_t                              max_seq_len,
                          size_t                              head_num,
                          size_t                              size_per_head,
                          size_t                              inter_size,
                          size_t                              num_layer,
                          size_t                              position_buckets,
                          float                               q_scaling,
                          cudaStream_t                        stream,
                          cublasMMWrapper*                    cublas_wrapper,
                          IAllocator*                         allocator,
                          bool                                is_free_buffer_after_forward):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    position_buckets_(position_buckets),
    q_scaling_(q_scaling)
{
    initialize();
}

template<typename T>
ZppEncoder<T>::~ZppEncoder()
{
    delete disentangled_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ZppEncoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ZppEncoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_         = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    attention_mask_         = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);

    deberta_emb_buf_ =
        (T*)allocator_->reMalloc(deberta_emb_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    deberta_in_buffer_ =
        (T*)allocator_->reMalloc(deberta_in_buffer_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    deberta_out_buffer_ =
        (T*)allocator_->reMalloc(deberta_out_buffer_, sizeof(T) * batch_size * seq_len * hidden_units_, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void ZppEncoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&h_pinned_token_num_ptr_), true);
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&attention_mask_));
        allocator_->free((void**)(&deberta_emb_buf_));
        allocator_->free((void**)(&deberta_in_buffer_));
        allocator_->free((void**)(&attn_out_buf_));
        allocator_->free((void**)(&deberta_out_buffer_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
void ZppEncoder<T>::forward(std::vector<Tensor>*       output_tensors,
                         const std::vector<Tensor>*    input_tensors,
                         const std::vector<Tensor>*    pos_query_cache,
                         const std::vector<Tensor>*    pos_key_cache,
                         const ZppEncoderWeight<T>*    deberta_weights)
{
    TensorMap input_tensors_map = TensorMap({{"input_ids", input_tensors->at(0)}, {"sequence_lengths", input_tensors->at(1)}});
    for (uint l = 0; l < num_layer_; l++) {
        input_tensors_map.insert("pos_query_cache_" + std::to_string(l), pos_query_cache->at(l));
        input_tensors_map.insert("pos_key_cache_" + std::to_string(l), pos_key_cache->at(l));
    }
    TensorMap output_tensors_map = TensorMap({{"output_hidden_state", output_tensors->at(0)}});
    forward(&output_tensors_map, &input_tensors_map, deberta_weights);
}

template<typename T>
void ZppEncoder<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors, const ZppEncoderWeight<T>* deberta_weights)
{
    // input_tensors:
    //      input_ids [batch, seqlen]
    //      sequence_lengths [batch]
    //      pos_query_cache [[batch, num_heads, 2 * position_buckets, head_size]]
    //      pos_key_cache [[batch, num_heads, 2 * position_buckets, head_size]]
    // output tensors:
    //      output_hidden_state [batch, seqlen, hidden]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 2);
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t request_seq_len    = input_tensors->at("input_ids").shape[1];
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("sequence_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->at("pos_query_cache_0").shape.size() == 4);
    FT_CHECK(input_tensors->at("pos_key_cache_0").shape.size() == 4);
    FT_CHECK(request_batch_size == input_tensors->at("sequence_lengths").shape[0]);
    FT_CHECK(request_batch_size == input_tensors->at("pos_query_cache_0").shape[0]);
    allocateBuffer(request_batch_size, request_seq_len);

    const int* input_ids        = input_tensors->at("input_ids").getPtr<int>();
    const int* sequence_lengths = input_tensors->at("sequence_lengths").getPtr<int>();

    DataType     data_type                 = getTensorType<T>();
    Tensor*      padding_offset_tensor_ptr = nullptr;
    size_t       h_token_num               = 0;

    T* deberta_input_ptr;
    T* deberta_output_ptr;

    // Word embedding layer [batch_size, seq_len] --> [batch_size, seq_len, hidden_size]
    invokeInputIdsWordEmbeddingLookup(
        deberta_emb_buf_,
        deberta_weights->word_embedding_table,
        input_ids,
        request_seq_len,
        request_seq_len,
        request_batch_size,
        hidden_units_,
        stream_
    );
    sync_check_cuda_error();

    //// Padding removal start
    
    // build attention mask from seq len
    invokeBuildEncoderAttentionMask(
        attention_mask_,
        sequence_lengths,
        request_batch_size,
        request_seq_len,
        stream_);
    sync_check_cuda_error();

    // compute cumulative number of word tokens & padded tokens
    invokeGetPaddingOffset(h_pinned_token_num_ptr_,
                            &h_token_num,
                            padding_offset_,
                            sequence_lengths,
                            request_batch_size,
                            request_seq_len,
                            stream_);

    // full input embeddings --> padding-entries-removed input embeddings
    invokeRemovePadding(
        deberta_in_buffer_, deberta_emb_buf_, padding_offset_, h_token_num, head_num_ * size_per_head_, stream_);
    sync_check_cuda_error();

    padding_offset_tensor_ptr =
        new Tensor(MEMORY_GPU, TYPE_INT32, std::vector<size_t>{h_token_num}, padding_offset_);
    
    //// Padding removal done

    // LayerNorm on word embeddings (do this after padding removing is better)
    invokeGeneralLayerNorm(deberta_in_buffer_,
                            deberta_in_buffer_,
                            deberta_weights->word_embedding_layernorm_weights.gamma,
                            deberta_weights->word_embedding_layernorm_weights.beta,
                            layernorm_eps_,
                            h_token_num,
                            hidden_units_,
                            (float*)nullptr,
                            0,
                            stream_);

    deberta_input_ptr       = deberta_in_buffer_;
    deberta_output_ptr      = deberta_out_buffer_;
    sync_check_cuda_error();

    // Encoder layers
    for (uint l = 0; l < num_layer_; l++) {
            T*                          from_tensor  = l == 0 ? deberta_input_ptr : deberta_output_ptr;
            T*                          out_tensor   = deberta_output_ptr;
        const ZppEncoderLayerWeight<T>& layer_weight = deberta_weights->deberta_layer_weights[l];

        // Attention
        {
            TensorMap attn_input_tensors{
                {"input_query",
                    Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{h_token_num, hidden_units_},
                        from_tensor}},
                {"attention_mask",
                    Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{request_batch_size, 1, request_seq_len, request_seq_len},
                        attention_mask_}},
                {"pos_query_cache",
                    Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{request_batch_size, head_num_, 2 * position_buckets_, size_per_head_},
                        input_tensors->at("pos_query_cache_" + std::to_string(l)).getPtr<T>()}},
                {"pos_key_cache",
                    Tensor{MEMORY_GPU,
                        data_type,
                        std::vector<size_t>{request_batch_size, head_num_, 2 * position_buckets_, size_per_head_},
                        input_tensors->at("pos_key_cache_" + std::to_string(l)).getPtr<T>()}}
            };
            
            attn_input_tensors.insertIfValid("padding_offset", *padding_offset_tensor_ptr);

            TensorMap attn_output_tensors{
                {"hidden_features",
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, attn_out_buf_}}};

            disentangled_attention_layer_->forward(
                &attn_output_tensors, &attn_input_tensors, &layer_weight.attention_weights);
        }

        invokeAddBiasResidualLayerNorm(attn_out_buf_,
                                        from_tensor,
                                        layer_weight.attention_weights.attention_output_weight.bias,
                                        layer_weight.attn_layernorm_weights.gamma,
                                        layer_weight.attn_layernorm_weights.beta,
                                        layernorm_eps_,
                                        h_token_num,
                                        hidden_units_,
                                        stream_);
        sync_check_cuda_error();

        // FFN (Intermediate + Output)
        {
            TensorMap ffn_input_tensors(
                {{"ffn_input",
                    Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{h_token_num, hidden_units_},
                            attn_out_buf_}}});
            TensorMap ffn_output_tensors(
                {{"ffn_output",
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, out_tensor}}});
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight.ffn_weights);
        }

        invokeAddBiasResidualLayerNorm(out_tensor,
                                        attn_out_buf_,
                                        layer_weight.ffn_weights.output_weight.bias,
                                        layer_weight.ffn_layernorm_weights.gamma,
                                        layer_weight.ffn_layernorm_weights.beta,
                                        layernorm_eps_,
                                        h_token_num,
                                        hidden_units_,
                                        stream_);
        sync_check_cuda_error();

    }  // transformer layers

    // post process (rebuild padding)
    invokeRebuildPadding(output_tensors->at("output_hidden_state").getPtr<T>(),
                            deberta_out_buffer_,
                            padding_offset_,
                            h_token_num,
                            head_num_ * size_per_head_,
                            stream_);
    sync_check_cuda_error();

    if (padding_offset_tensor_ptr != nullptr) {
        delete padding_offset_tensor_ptr;
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    cudaStreamSynchronize(stream_);
}

template class ZppEncoder<half>;

}  // namespace fastertransformer
