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

#include "src/fastertransformer/models/zcode_encoder/Deberta.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"

namespace fastertransformer {

template<typename T>
void ZcodeEncoder<T>::initialize()
{
    disentangled_attention_layer_ =
        new ZcodeEncoderDisentangledAttentionLayer<T>(0,
                                                        0,
                                                        head_num_,
                                                        size_per_head_,
                                                        relative_position_buckets_,
                                                        head_num_ * size_per_head_,
                                                        q_scaling_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_,
                                                        sparse_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new GeluFfnLayer<T>(0,
                                        0,
                                        head_num_,
                                        size_per_head_,
                                        0,  // expert_num
                                        inter_size_,
                                        stream_,
                                        cublas_wrapper_,
                                        allocator_,
                                        is_free_buffer_after_forward_,
                                        sparse_,
                                        0,
                                        use_gated_activation);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new ReluFfnLayer<T>(0,
                                        0,
                                        head_num_,
                                        size_per_head_,
                                        0,  // expert_num
                                        inter_size_,
                                        stream_,
                                        cublas_wrapper_,
                                        allocator_,
                                        is_free_buffer_after_forward_,
                                        sparse_,
                                        0,
                                        use_gated_activation);
    }
}

template<typename T>
ZcodeEncoder<T>::ZcodeEncoder(size_t                              max_batch_size,
                    size_t                              max_seq_len,
                    size_t                              head_num,
                    size_t                              size_per_head,
                    size_t                              max_relative_positions,
                    size_t                              relative_position_buckets,
                    size_t                              inter_size,
                    size_t                              num_layer,
                    float                               q_scaling,
                    cudaStream_t                        stream,
                    cublasMMWrapper*                    cublas_wrapper,
                    IAllocator*                         allocator,
                    bool                                is_free_buffer_after_forward,
                    bool                                sparse,
                    ActivationType                      activation_type,
                    LayerNormType                       layernorm_type,
                    NcclParam                           tensor_para,
                    NcclParam                           pipeline_para,
                    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                    bool                                enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    head_num_(head_num),
    size_per_head_(size_per_head),
    max_relative_positions_(max_relative_positions),
    relative_position_buckets_(relative_position_buckets),
    inter_size_(inter_size),
    hidden_units_(head_num_ * size_per_head_),
    num_layer_(num_layer),
    q_scaling_(q_scaling),
    sparse_(sparse),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
ZcodeEncoder<T>::ZcodeEncoder(size_t           max_batch_size,
                    size_t           max_seq_len,
                    size_t           head_num,
                    size_t           size_per_head,
                    size_t           max_relative_positions,
                    size_t           relative_position_buckets,
                    size_t           inter_size,
                    size_t           num_layer,
                    float            q_scaling,
                    cudaStream_t     stream,
                    cublasMMWrapper* cublas_wrapper,
                    IAllocator*      allocator,
                    bool             is_free_buffer_after_forward,
                    bool             sparse,
                    ActivationType   activation_type,
                    LayerNormType    layernorm_type):
    ZcodeEncoder(max_batch_size,
            max_seq_len,
            head_num,
            size_per_head,
            max_relative_positions,
            relative_position_buckets,
            inter_size,
            num_layer,
            q_scaling,
            stream,
            cublas_wrapper,
            allocator,
            is_free_buffer_after_forward,
            sparse,
            activation_type,
            layernorm_type,
            NcclParam(0, 1),
            NcclParam(0, 1),
            nullptr,
            false)
{
}

template<typename T>
ZcodeEncoder<T>::ZcodeEncoder(ZcodeEncoder<T> const& deberta):
    ZcodeEncoder(0,
            0,
            deberta.head_num_,
            deberta.size_per_head_,
            deberta.max_relative_positions_,
            deberta.relative_position_buckets_,
            deberta.inter_size_,
            deberta.num_layer_,
            deberta.q_scaling_,
            deberta.stream_,
            deberta.cublas_wrapper_,
            deberta.allocator_,
            deberta.is_free_buffer_after_forward_,
            deberta.sparse_,
            deberta.activation_type_,
            deberta.layernorm_type_,
            deberta.tensor_para_,
            deberta.pipeline_para_,
            deberta.custom_all_reduce_comm_,
            deberta.enable_custom_all_reduce_)
{
}

template<typename T>
ZcodeEncoder<T>::~ZcodeEncoder()
{
    delete disentangled_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ZcodeEncoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ZcodeEncoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    h_pinned_token_num_ptr_ = (size_t*)allocator_->reMalloc(h_pinned_token_num_ptr_, sizeof(size_t), true, true);
    padding_offset_         = (int*)allocator_->reMalloc(padding_offset_, sizeof(int) * batch_size * seq_len, false);
    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);

    deberta_emb_buf_ =
        (T*)allocator_->reMalloc(deberta_emb_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    deberta_in_buffer_ =
        (T*)allocator_->reMalloc(deberta_in_buffer_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    deberta_out_buffer_ =
        (T*)allocator_->reMalloc(deberta_out_buffer_, sizeof(T) * batch_size * seq_len * hidden_units_, false);

    if (layernorm_type_ == LayerNormType::post_layernorm) {
        normed_from_tensor_  = nullptr;
        normed_attn_out_buf_ = nullptr;
    }
    else {
        normed_from_tensor_ =
            (T*)allocator_->reMalloc(normed_from_tensor_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
        normed_attn_out_buf_ =
            (T*)allocator_->reMalloc(normed_attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void ZcodeEncoder<T>::freeBuffer()
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

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            normed_from_tensor_  = nullptr;
            normed_attn_out_buf_ = nullptr;
        }
        else {
            allocator_->free((void**)(&normed_from_tensor_));
            allocator_->free((void**)(&normed_attn_out_buf_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ZcodeEncoder<T>::isValidLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ZcodeEncoder<T>::isFirstLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ZcodeEncoder<T>::isLastLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ZcodeEncoder<T>::getFirstLayerParallelId()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
void ZcodeEncoder<T>::forward(std::vector<Tensor>*       output_tensors,
                         const std::vector<Tensor>* input_tensors,
                         const TensorMap*           pos_query_cache,
                         const TensorMap*           pos_key_cache,
                         const ZcodeEncoderWeight<T>*    deberta_weights)
{
    TensorMap input_tensors_map =
        TensorMap({{"input_ids", input_tensors->at(0)}, {"sequence_lengths", input_tensors->at(1)}});
    TensorMap output_tensors_map = TensorMap({{"output_hidden_state", output_tensors->at(0)}});
    forward(&output_tensors_map, &input_tensors_map, pos_query_cache, pos_key_cache, deberta_weights);
}

template<typename T>
void ZcodeEncoder<T>::forward(
    TensorMap* output_tensors, 
    TensorMap* input_tensors, 
    const TensorMap* pos_query_cache,
    const TensorMap* pos_key_cache,
    const ZcodeEncoderWeight<T>* deberta_weights
)
{
    // input_tensors:
    //      input_ids [batch, seqlen]
    //      sequence_lengths [batch]
    // output tensors:
    //      output_hidden_state [batch, seqlen, hidden]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 2);
    const size_t request_batch_size = input_tensors->at("input_ids").shape[0];
    const size_t request_seq_len    = input_tensors->at("input_ids").shape[1];
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("sequence_lengths").shape.size() == 1);
    FT_CHECK(request_batch_size == input_tensors->at("sequence_lengths").shape[0]);
    allocateBuffer(request_batch_size, request_seq_len);

    const int* sequence_lengths = input_tensors->at("sequence_lengths").getPtr<int>();

    DataType     data_type        = getTensorType<T>();
    const size_t local_batch_size = getLocalBatchSize(request_batch_size, request_seq_len, pipeline_para_.world_size_);
    FT_CHECK(request_batch_size % local_batch_size == 0);
    const size_t iteration_num = request_batch_size / local_batch_size;

    for (uint ite = 0; ite < iteration_num; ite++) {
        Tensor*      padding_offset_tensor_ptr = nullptr;
        size_t       id_offset                 = ite * local_batch_size;
        const size_t hidden_offset             = ite * local_batch_size * request_seq_len * hidden_units_;
        size_t       h_token_num               = 0;

        T* deberta_input_ptr;
        T* deberta_output_ptr;

        // Word embedding layer [batch_size, seq_len] --> [batch_size, seq_len, hidden_size]
        invokeInputIdsEmbeddingLookupPosEncoding(
            deberta_emb_buf_,
            nullptr,
            deberta_weights->word_embedding_table,
            (T*)nullptr,  // word embedding only, position embedding was replaced by relative embedding design in
                          // DeBERTa
            pPromptTuningParam<T>{},
            input_tensors->at("input_ids").getPtrWithOffset<int>(id_offset * request_seq_len),
            1,
            request_seq_len,
            request_seq_len,
            local_batch_size,
            hidden_units_,
            stream_);
        sync_check_cuda_error();

        //// Padding removal
        // build attention mask from seq len
        invokeBuildEncoderAttentionMask(
            attention_mask_,
            input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
            local_batch_size,
            request_seq_len,
            stream_);
        sync_check_cuda_error();

        // compute cumulative number of word tokens & padded tokens
        invokeGetPaddingOffset(h_pinned_token_num_ptr_,
                               &h_token_num,
                               padding_offset_,
                               input_tensors->at("sequence_lengths").getPtrWithOffset<int>(ite * local_batch_size),
                               local_batch_size,
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

        deberta_input_ptr               = deberta_in_buffer_;
        deberta_output_ptr              = deberta_out_buffer_;
        sync_check_cuda_error();

        // Encoder layers
        for (uint l = 0; l < num_layer_; l++) {
            if (!isValidLayerParallelId(l)) {
                continue;
            }
            T*                           from_tensor  = l == 0 ? deberta_input_ptr : deberta_output_ptr;
            T*                           out_tensor   = deberta_output_ptr;
            const ZcodeEncoderLayerWeight<T>& layer_weight = deberta_weights->deberta_layer_weights[l];

            if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
                ftNcclRecv(from_tensor + h_token_num * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                           h_token_num * hidden_units_ / tensor_para_.world_size_,
                           pipeline_para_.rank_ - 1,
                           pipeline_para_,
                           stream_);
                if (tensor_para_.world_size_ > 1) {
                    ftNcclAllGather(from_tensor,
                                    from_tensor,
                                    h_token_num * hidden_units_ / tensor_para_.world_size_,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                }
            }
            if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralLayerNorm(normed_from_tensor_,
                                       from_tensor,
                                       layer_weight.attn_layernorm_weights.gamma,
                                       layer_weight.attn_layernorm_weights.beta,
                                       layernorm_eps_,
                                       h_token_num,
                                       hidden_units_,
                                       (float*)nullptr,
                                       0,
                                       stream_);
                sync_check_cuda_error();
            }

            // Attention
            {
                TensorMap attn_input_tensors{
                    {"input_query",
                     Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{h_token_num, hidden_units_},
                            layernorm_type_ == LayerNormType::pre_layernorm ? normed_from_tensor_ : from_tensor}},
                    {"attention_mask",
                     Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{local_batch_size, 1, request_seq_len, request_seq_len},
                            attention_mask_}},
                    
                    {"pos_query_cache", pos_query_cache->at("pos_query_cache_" + std::to_string(l))},
                    {"pos_key_cache", pos_key_cache->at("pos_key_cache_" + std::to_string(l))},
                };
                attn_input_tensors.insertIfValid("padding_offset", *padding_offset_tensor_ptr);

                TensorMap attn_output_tensors{
                    {"hidden_features",
                     Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, attn_out_buf_}}};

                disentangled_attention_layer_->forward(
                    &attn_output_tensors, &attn_input_tensors, &layer_weight.attention_weights);
            }

            if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(attn_out_buf_,
                                               from_tensor,
                                               layer_weight.attention_weights.attention_output_weight.bias,
                                               layer_weight.attn_layernorm_weights.gamma,
                                               layer_weight.attn_layernorm_weights.beta,
                                               layernorm_eps_,
                                               h_token_num,
                                               hidden_units_,
                                               stream_);
            }
            else if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeGeneralAddBiasResidualPreLayerNorm(attn_out_buf_,
                                                         normed_attn_out_buf_,
                                                         attn_out_buf_,
                                                         from_tensor,
                                                         layer_weight.ffn_layernorm_weights.gamma,
                                                         layer_weight.ffn_layernorm_weights.beta,
                                                         layer_weight.attention_weights.attention_output_weight.bias,
                                                         layernorm_eps_,
                                                         h_token_num,
                                                         hidden_units_,
                                                         (float*)nullptr,
                                                         (float*)nullptr,
                                                         (float*)nullptr,
                                                         (float*)nullptr,
                                                         0,
                                                         stream_);
            }
            sync_check_cuda_error();

            // FFN (Intermediate + Output)
            {
                TensorMap ffn_input_tensors(
                    {{"ffn_input",
                      Tensor{MEMORY_GPU,
                             data_type,
                             std::vector<size_t>{h_token_num, hidden_units_},
                             layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}}});
                TensorMap ffn_output_tensors(
                    {{"ffn_output",
                      Tensor{MEMORY_GPU, data_type, std::vector<size_t>{h_token_num, hidden_units_}, out_tensor}}});
                ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight.ffn_weights);
            }

            if (layernorm_type_ == LayerNormType::post_layernorm) {
                invokeAddBiasResidualLayerNorm(out_tensor,
                                               attn_out_buf_,
                                               layer_weight.ffn_weights.output_weight.bias,
                                               layer_weight.ffn_layernorm_weights.gamma,
                                               layer_weight.ffn_layernorm_weights.beta,
                                               layernorm_eps_,
                                               h_token_num,
                                               hidden_units_,
                                               stream_);
            }
            else if (layernorm_type_ == LayerNormType::pre_layernorm) {
                invokeAddBiasResidual(out_tensor,
                                      attn_out_buf_,
                                      layer_weight.ffn_weights.output_weight.bias,
                                      h_token_num,
                                      hidden_units_,
                                      stream_);
            }
            sync_check_cuda_error();

            if (isLastLayerParallelId(l) && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                && pipeline_para_.world_size_ > 1) {

                ftNcclSend(out_tensor + h_token_num * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                           h_token_num * hidden_units_ / tensor_para_.world_size_,
                           pipeline_para_.rank_ + 1,
                           pipeline_para_,
                           stream_);
            }
        }  // transformer layers

        if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
            // post process (rebuild padding)
            invokeRebuildPadding(output_tensors->at("output_hidden_state").getPtrWithOffset<T>(hidden_offset),
                                 deberta_out_buffer_,
                                 padding_offset_,
                                 h_token_num,
                                 head_num_ * size_per_head_,
                                 stream_);
            sync_check_cuda_error();
        }

        if (padding_offset_tensor_ptr != nullptr) {
            delete padding_offset_tensor_ptr;
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    if (pipeline_para_.world_size_ > 1) {
        ftNcclGroupStart();
        const int data_size = request_batch_size * request_seq_len * hidden_units_ / tensor_para_.world_size_;
        ftNcclBroadCast(output_tensors->at("output_hidden_state").getPtr<T>() + data_size * tensor_para_.rank_,
                        data_size,
                        pipeline_para_.world_size_ - 1,
                        pipeline_para_,
                        stream_);
        ftNcclGroupEnd();

        sync_check_cuda_error();
        if (tensor_para_.world_size_ > 1) {
            ftNcclAllGather(output_tensors->at("output_hidden_state").getPtr<T>(),
                            output_tensors->at("output_hidden_state").getPtr<T>(),
                            data_size,
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
        }
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
    }
    cudaStreamSynchronize(stream_);
}

template class ZcodeEncoder<float>;
template class ZcodeEncoder<half>;
#ifdef ENABLE_BF16
template class ZcodeEncoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
