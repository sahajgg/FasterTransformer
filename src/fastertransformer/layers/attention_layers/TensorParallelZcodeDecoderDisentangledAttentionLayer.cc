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

#include "src/fastertransformer/layers/attention_layers/TensorParallelZcodeDecoderDisentangledAttentionLayer.h"

namespace fastertransformer {

template<typename T>
void TensorParallelZcodeDecoderDisentangledAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                                          TensorMap*                input_tensors,
                                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, d_model],
    //      attention_mask [batch, 1, seqlen, seqlen],
    //      relative_embeddings [2*attention_span, hidden_size],
    //      padding_offset [token_num] (optional)
    //
    // output_tensors:
    //      hidden_features [token_num, d_model]
    //
    // For more information, please refer to ZcodeDecoderDisentangledAttentionLayer

    const size_t size = output_tensors->at("hidden_features").size();
    std::vector<Tensor> hidden_features_reduce = {output_tensors->at("hidden_features")};

    bool use_custom_all_reduce_kernel = false;
    if (enable_custom_all_reduce_ && custom_all_reduce_comm_ != nullptr) {
        use_custom_all_reduce_kernel = custom_all_reduce_comm_->swapInternalBuffer(&hidden_features_reduce, size);
        output_tensors->at("hidden_features").data = hidden_features_reduce[0].data;
    }

    ZcodeDecoderDisentangledAttentionLayer<T>::forward(output_tensors, input_tensors, attention_weights);

    T* attention_out = output_tensors->getPtr<T>("hidden_features");
    if (tensor_para_.world_size_ > 1) {
        if (!use_custom_all_reduce_kernel) {
            ftNcclAllReduceSum(
                attention_out, attention_out, size, tensor_para_, ZcodeDecoderDisentangledAttentionLayer<T>::stream_);
        }
        else {
            custom_all_reduce_comm_->customAllReduce(size, ZcodeDecoderDisentangledAttentionLayer<T>::stream_);
            output_tensors->at("hidden_features").data = hidden_features_reduce[0].data;
        }
        sync_check_cuda_error();
    }
}

template<typename T>
TensorParallelZcodeDecoderDisentangledAttentionLayer<T>::TensorParallelZcodeDecoderDisentangledAttentionLayer(
    size_t                              max_batch_size,
    size_t                              max_seq_len,
    size_t                              head_num,
    size_t                              size_per_head,
    size_t                              attention_span,
    size_t                              d_model,
    float                               q_scaling,
    NcclParam                           tensor_para,
    cudaStream_t                        stream,
    cublasMMWrapper*                    cublas_wrapper,
    IAllocator*                         allocator,
    bool                                is_free_buffer_after_forward,
    bool                                is_sparse,
    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
    int                                 enable_custom_all_reduce):
    ZcodeDecoderDisentangledAttentionLayer<T>(max_batch_size,
                                  max_seq_len,
                                  head_num / tensor_para.world_size_,
                                  size_per_head,
                                  attention_span,
                                  d_model,
                                  q_scaling,
                                  stream,
                                  cublas_wrapper,
                                  allocator,
                                  is_free_buffer_after_forward,
                                  is_sparse),
    tensor_para_(tensor_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    FT_CHECK(head_num % tensor_para_.world_size_ == 0);
}

template<typename T>
TensorParallelZcodeDecoderDisentangledAttentionLayer<T>::TensorParallelZcodeDecoderDisentangledAttentionLayer(
    TensorParallelZcodeDecoderDisentangledAttentionLayer<T> const& attention_layer):
    ZcodeDecoderDisentangledAttentionLayer<T>(attention_layer), tensor_para_(attention_layer.tensor_para_)
{
}

template class TensorParallelZcodeDecoderDisentangledAttentionLayer<float>;
template class TensorParallelZcodeDecoderDisentangledAttentionLayer<half>;
#ifdef ENABLE_BF16
template class TensorParallelZcodeDecoderDisentangledAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer