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

#include "src/fastertransformer/models/zcode_decoder/Deberta.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"

namespace fastertransformer {

template<typename T>
void ZcodeDecoder<T>::initialize()
{
    disentangled_attention_layer_ =
        new ZcodeDecoderDisentangledAttentionLayer<T>(0,
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

    cross_attention_layer_ = new DecoderCrossAttentionLayer<T>(0,
                                                        head_num_,
                                                        size_per_head_,
                                                        head_num_ * size_per_head_,
                                                        sqrtf(q_scaling_),
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU;
    if (activation_type_ == ActivationType::Gelu) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(0,
                                                       0,
                                                       head_num_,
                                                       size_per_head_,
                                                       0,  // expert_num
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(0,
                                                       0,
                                                       head_num_,
                                                       size_per_head_,
                                                       0,  // expert_num
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       sparse_,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
ZcodeDecoder<T>::ZcodeDecoder(size_t                              max_batch_size,
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
ZcodeDecoder<T>::ZcodeDecoder(size_t           max_batch_size,
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
    ZcodeDecoder(max_batch_size,
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
ZcodeDecoder<T>::ZcodeDecoder(ZcodeDecoder<T> const& deberta):
    ZcodeDecoder(0,
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
ZcodeDecoder<T>::~ZcodeDecoder()
{
    delete disentangled_attention_layer_;
    delete cross_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ZcodeDecoder<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ZcodeDecoder<T>::allocateBuffer(size_t batch_size, size_t seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    attention_mask_ = (T*)allocator_->reMalloc(attention_mask_, sizeof(T) * batch_size * seq_len * seq_len, false);

    deberta_in_buffer_ =
        (T*)allocator_->reMalloc(deberta_in_buffer_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    attn_out_buf_ = (T*)allocator_->reMalloc(attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    cross_attn_out_buf_ =
        (T*)allocator_->reMalloc(cross_attn_out_buf_, sizeof(T) * batch_size * seq_len * hidden_units_, false);
    decoder_layer_output_ =
        (T*)allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * seq_len * hidden_units_, false);

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
void ZcodeDecoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&attention_mask_));
        allocator_->free((void**)(&deberta_in_buffer_));
        allocator_->free((void**)(&attn_out_buf_));
        allocator_->free((void**)(&cross_attn_out_buf_));
        allocator_->free((void**)(&decoder_layer_output_));

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
bool ZcodeDecoder<T>::isValidLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ZcodeDecoder<T>::isFirstLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ZcodeDecoder<T>::isLastLayerParallelId(uint l)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ZcodeDecoder<T>::getFirstLayerParallelId()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
void ZcodeDecoder<T>::forward(std::vector<Tensor>*         output_tensors,
                         const std::vector<Tensor>*        input_tensors,
                         const TensorMap*                  pos_query_cache,
                         const TensorMap*                  pos_key_cache,
                         const std::vector<ZcodeDecoderLayerWeight<T>>* deberta_layer_weights)
{
    TensorMap input_tensors_map = TensorMap({
            {"decoder_input", input_tensors->at(0)},
            {"current_cache_seq_len", input_tensors->at(1)},
            {"encoder_output", input_tensors->at(2)},
            {"encoder_sequence_length", input_tensors->at(3)},
            {"finished", input_tensors->at(4)},
            {"step", input_tensors->at(5)},
            {"ite", input_tensors->at(6)},
            {"cache_indirection", input_tensors->at(7)},
            {"attention_mask", input_tensors->at(8)}
    });
    TensorMap output_tensors_map = TensorMap({
            {"decoder_output", output_tensors->at(0)},
            {"key_cache", output_tensors->at(1)},
            {"value_cache", output_tensors->at(2)},
            {"key_mem_cache", output_tensors->at(3)},
            {"value_mem_cache", output_tensors->at(4)}
    });
    forward(&output_tensors_map, &input_tensors_map, pos_query_cache, pos_key_cache, deberta_layer_weights);
}

template<typename T>
void ZcodeDecoder<T>::forward(
    TensorMap*                        output_tensors, 
    TensorMap*                        input_tensors, 
    const TensorMap*                  pos_query_cache,
    const TensorMap*                  pos_key_cache,
    const std::vector<ZcodeDecoderLayerWeight<T>>* deberta_layer_weights
)
{
    // input tensors:
    //      decoder_input [request_batch_size, request_seq_len, d_model_],
    //      current_cache_seq_len [1]
    //      encoder_output [request_batch_size, mem_max_seq_len, mem_d_model_],
    //      encoder_sequence_length [request_batch_size],
    //      finished [request_batch_size],
    //      step [1] on cpu
    //      ite [1] on cpu
    //      cache_indirection [request_batch_size / beam_width, beam_width, max_seq_len]
    //      (NOTE: Here, request_batch_size contains the beam_width, so request_batch_size / beam_width)
    //      attention_mask [request_batch_size, 1, max_seq_len, max_seq_len]

    // output tensors:
    //      decoder_output [request_batch_size, d_model_],
    //      key_cache [num_layer / pipeline_para_.world_size_, request_batch_size, head_num, max_seq_len, size_per_head]
    //      value_cache [num_layer / pipeline_para_.world_size_, request_batch_size, head_num, max_seq_len, size_per_head]
    //      key_mem_cache [num_layer / pipeline_para_.world_size_, request_batch_size, mem_max_seq_len, hidden_dimension],
    //      value_mem_cache [num_layer / pipeline_para_.world_size_, request_batch_size, mem_max_seq_len, hidden_dimension]

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() == 8 || input_tensors->size() == 9);
    FT_CHECK(output_tensors->size() == 5);
    const size_t request_batch_size = input_tensors->at("decoder_input").shape[0];
    const size_t request_seq_len    = input_tensors->at("decoder_input").shape[1];
    const size_t mem_max_seq_len    = input_tensors->at("encoder_output").shape[1];
    
    FT_CHECK(input_tensors->at("decoder_input").shape.size() == 3);
    allocateBuffer(request_batch_size, request_seq_len);

    const uint     ite             = input_tensors->at("ite").getVal<uint>();
    const int      step            = input_tensors->at("step").getVal<int>();
    DataType       data_type       = getTensorType<T>();

    std::vector<size_t> self_k_cache_shape;
    self_k_cache_shape.push_back(request_batch_size);
    for (auto t = output_tensors->at("key_cache").shape.begin() + 2; t != output_tensors->at("key_cache").shape.end(); ++t) {
        self_k_cache_shape.push_back(*t);
    }
    std::vector<size_t> self_v_cache_shape;
    self_v_cache_shape.push_back(request_batch_size);
    for (auto t = output_tensors->at("value_cache").shape.begin() + 2; t != output_tensors->at("value_cache").shape.end(); ++t) {
        self_v_cache_shape.push_back(*t);
    }

    const std::vector<size_t> mem_cache_shape = {request_batch_size, output_tensors->at("key_mem_cache").shape[2], output_tensors->at("key_mem_cache").shape[3]};

    // Decoder layers
    for (uint l = 0; l < num_layer_; l++) {
        if (!isValidLayerParallelId(l)) {
            continue;
        }
        T* decoder_input  = l == 0 ? input_tensors->at("decoder_input").getPtr<T>() : decoder_layer_output_;
        T* decoder_output = (l == num_layer_ - 1) ? output_tensors->at("decoder_output").getPtr<T>() : decoder_layer_output_;
        const ZcodeDecoderLayerWeight<T>& layer_weight = deberta_layer_weights->at(l);

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            ftNcclRecv(decoder_input + request_batch_size * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       request_batch_size * hidden_units_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(decoder_input,
                                decoder_input,
                                (int)request_batch_size * hidden_units_ / tensor_para_.world_size_,
                                tensor_para_.rank_,
                                tensor_para_,
                                stream_);
            }
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = output_tensors->at("key_cache").shape.begin() + 1; t != output_tensors->at("key_cache").shape.end(); ++t) {
            cache_offset *= *t;
        };

        size_t mem_cache_offset = l - getFirstLayerParallelId();
        for (auto t = output_tensors->at("key_mem_cache").shape.begin() + 1; t != output_tensors->at("key_mem_cache").shape.end(); ++t) {
            mem_cache_offset *= *t;
        };
        
        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralLayerNorm(normed_from_tensor_,
                                    decoder_input,
                                    layer_weight.attn_layernorm_weights.gamma,
                                    layer_weight.attn_layernorm_weights.beta,
                                    layernorm_eps_,
                                    request_batch_size * request_seq_len,
                                    hidden_units_,
                                    (float*)nullptr,
                                    0,
                                    stream_);
            sync_check_cuda_error();
        }
    
        // Attention
        {
            TensorMap attn_input_tensors{
                {"input_query", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{request_batch_size, request_seq_len, hidden_units_},
                        layernorm_type_ == LayerNormType::pre_layernorm ? normed_from_tensor_ : decoder_input}},
                {"pos_query_cache", pos_query_cache->at("pos_query_cache_" + std::to_string(l))},
                {"pos_key_cache", pos_key_cache->at("pos_key_cache_" + std::to_string(l))},
                {"current_cache_seq_len", input_tensors->at("current_cache_seq_len")},
                {"attention_mask", input_tensors->at("attention_mask")}
            };
            
            if(input_tensors->isExist("cache_indirection")){
                attn_input_tensors.insertIfValid("cache_indirection", input_tensors->at("cache_indirection"));
            }

            TensorMap attn_output_tensors{
                {"hidden_features",
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{request_batch_size, request_seq_len, hidden_units_}, attn_out_buf_}},
                {"key_cache",
                    Tensor{MEMORY_GPU, data_type, self_k_cache_shape, output_tensors->at("key_cache").getPtrWithOffset(cache_offset)}},
                {"value_cache",
                    Tensor{MEMORY_GPU, data_type, self_v_cache_shape, output_tensors->at("value_cache").getPtrWithOffset(cache_offset)}}
            };

            disentangled_attention_layer_->forward(
                &attn_output_tensors, &attn_input_tensors, &layer_weight.attention_weights);
        }
        sync_check_cuda_error();

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeAddBiasResidualLayerNorm(attn_out_buf_,
                                            decoder_input,
                                            layer_weight.attention_weights.attention_output_weight.bias,
                                            layer_weight.attn_layernorm_weights.gamma,
                                            layer_weight.attn_layernorm_weights.beta,
                                            layernorm_eps_,
                                            request_batch_size * request_seq_len,
                                            hidden_units_,
                                            stream_);
        }
        else if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralAddBiasResidualPreLayerNorm(attn_out_buf_,
                                                        normed_attn_out_buf_,
                                                        attn_out_buf_,
                                                        decoder_input,
                                                        layer_weight.ffn_layernorm_weights.gamma,
                                                        layer_weight.ffn_layernorm_weights.beta,
                                                        layer_weight.attention_weights.attention_output_weight.bias,
                                                        layernorm_eps_,
                                                        request_batch_size * request_seq_len,
                                                        hidden_units_,
                                                        (float*)nullptr,
                                                        (float*)nullptr,
                                                        (float*)nullptr,
                                                        (float*)nullptr,
                                                        0,
                                                        stream_);
        }
        sync_check_cuda_error();

        // std::cout << "attn_out_buf_" << std::endl;
        // print_to_screen(attn_out_buf_, 10);
        // print_to_screen(attn_out_buf_ + (request_seq_len * hidden_units_), 10);
        // std::cout << std::endl;

        // Cross Attention
        {
            TensorMap cross_attention_input_tensors{
                {"input_query", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{request_batch_size * request_seq_len, hidden_units_}, 
                layernorm_type_ == LayerNormType::pre_layernorm ? normed_attn_out_buf_ : attn_out_buf_}},
                {"encoder_output", input_tensors->at("encoder_output")},
                {"encoder_sequence_length", input_tensors->at("encoder_sequence_length")},
                {"finished", input_tensors->at("finished")},
                {"step", input_tensors->at("step")}
            };

            TensorMap cross_attention_output_tensors{
                {"hidden_features", 
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{request_batch_size * request_seq_len, hidden_units_}, cross_attn_out_buf_}},
                {"key_cache",
                    Tensor{MEMORY_GPU, data_type, mem_cache_shape, output_tensors->at("key_mem_cache").getPtrWithOffset(mem_cache_offset)}},
                {"value_cache",
                    Tensor{MEMORY_GPU, data_type, mem_cache_shape, output_tensors->at("value_mem_cache").getPtrWithOffset(mem_cache_offset)}}
            };
            
            cross_attention_layer_->forward(&cross_attention_output_tensors, &cross_attention_input_tensors, &layer_weight.cross_attention_weights);
        }

        invokeAddBiasResidualLayerNorm(cross_attn_out_buf_,
                                        attn_out_buf_,
                                        layer_weight.cross_attention_weights.attention_output_weight.bias,
                                        layer_weight.cross_attn_layernorm_weights.gamma,
                                        layer_weight.cross_attn_layernorm_weights.beta,
                                        layernorm_eps_,
                                        request_batch_size * request_seq_len,
                                        hidden_units_,
                                        stream_);

        // FFN (Intermediate + Output)
        {
            TensorMap ffn_input_tensors(
                {{"ffn_input",
                    Tensor{MEMORY_GPU,
                            data_type,
                            std::vector<size_t>{request_batch_size * request_seq_len, hidden_units_}, cross_attn_out_buf_}}});
            TensorMap ffn_output_tensors(
                {{"ffn_output",
                    Tensor{MEMORY_GPU, data_type, std::vector<size_t>{request_batch_size * request_seq_len, hidden_units_}, decoder_output}}});
            ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &layer_weight.ffn_weights);
        }

        if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeAddBiasResidualLayerNorm(decoder_output,
                                            cross_attn_out_buf_,
                                            layer_weight.ffn_weights.output_weight.bias,
                                            layer_weight.ffn_layernorm_weights.gamma,
                                            layer_weight.ffn_layernorm_weights.beta,
                                            layernorm_eps_,
                                            request_batch_size * request_seq_len,
                                            hidden_units_,
                                            stream_);
        }
        else if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeAddBiasResidual(decoder_output,
                                    cross_attn_out_buf_,
                                    layer_weight.ffn_weights.output_weight.bias,
                                    request_batch_size * request_seq_len,
                                    hidden_units_,
                                    stream_);
        }
        sync_check_cuda_error();

        if (isLastLayerParallelId(l) && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1 && pipeline_para_.world_size_ > 1) {

            ftNcclSend(decoder_output + request_batch_size * request_seq_len * hidden_units_ / tensor_para_.world_size_ * tensor_para_.rank_,
                        request_batch_size * request_seq_len * hidden_units_ / tensor_para_.world_size_,
                        pipeline_para_.rank_ + 1,
                        pipeline_para_,
                        stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }  
    sync_check_cuda_error();

    if (pipeline_para_.world_size_ > 1) {
        ftNcclGroupStart();
        const int data_size = request_batch_size * request_seq_len * hidden_units_ / tensor_para_.world_size_;
        ftNcclBroadCast(output_tensors->at("decoder_output").getPtr<T>() + data_size * tensor_para_.rank_,
                        data_size,
                        pipeline_para_.world_size_ - 1,
                        pipeline_para_,
                        stream_);
        ftNcclGroupEnd();

        sync_check_cuda_error();
        if (tensor_para_.world_size_ > 1) {
            ftNcclAllGather(output_tensors->at("decoder_output").getPtr<T>(),
                            output_tensors->at("decoder_output").getPtr<T>(),
                            data_size,
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
        }
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
    }
}

template class ZcodeDecoder<float>;
template class ZcodeDecoder<half>;
#ifdef ENABLE_BF16
template class ZcodeDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
