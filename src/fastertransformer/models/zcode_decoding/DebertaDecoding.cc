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

#include "src/fastertransformer/models/zcode_decoding/DebertaDecoding.h"
#include "src/fastertransformer/kernels/beam_search_topk_kernels.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"

namespace fastertransformer {

template<typename T>
void DebertaDecoding<T>::initialize()
{
    decoder_ = new ZcodeDecoder<T>(0,  // max_batch_size_ * beam_width_,
                                0,
                                head_num_,
                                size_per_head_,
                                max_relative_positions_,
                                relative_position_buckets_,
                                inter_size_,
                                num_layer_,
                                q_scaling_,
                                stream_,
                                cublas_wrapper_,
                                allocator_,
                                is_free_buffer_after_forward_,
                                false,
                                activation_type_,
                                LayerNormType::post_layernorm,
                                NcclParam(0, 1),
                                NcclParam(0, 1),
                                nullptr,
                                false);

    dynamic_decode_layer_ = new DynamicDecodeLayer<DynamicDecodeType>(vocab_size_,
                                                                      vocab_size_,
                                                                      0,  // end_id, deprecated
                                                                      stream_,
                                                                      cublas_wrapper_,
                                                                      allocator_,
                                                                      is_free_buffer_after_forward_,
                                                                      cuda_device_prop_);
}

template<typename T>
void DebertaDecoding<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void DebertaDecoding<T>::allocateBuffer(
    size_t batch_size, size_t beam_width, size_t seq_len, size_t max_seq_len, size_t max_mem_seq_len, size_t encoder_d_model)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // Note: To put the start_ids, we use max_seq_len + 1 for ouptut_ids_buf_
    // And to consistent to the output_ids_buf_, some related buffers are also
    // use max_seq_len + 1, but not max_seq_len.
    // This only affects the buffer size, not affect the performance.

    const size_t batchxbeam      = batch_size * beam_width;
    const size_t self_cache_size = num_layer_ * batchxbeam * (max_seq_len + 1) * hidden_units_;
    const size_t mem_cache_size = num_layer_ * batchxbeam * max_mem_seq_len * hidden_units_;

    deberta_emb_buf_   = (T*)(allocator_->malloc(sizeof(T) * batchxbeam * seq_len * d_model_, false));
    decoder_input_buf_  = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * d_model_, false));
    decoder_output_buf_ = (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * d_model_, false));
    decoding_sequence_lengths_ = (int*)(allocator_->reMalloc(decoding_sequence_lengths_, sizeof(int) * batchxbeam, false));

    logits_buf_iter      = (T*)(allocator_->reMalloc(
        logits_buf_iter, sizeof(T) * batchxbeam * vocab_size_, false));
    logits_buf_      = (T*)(allocator_->reMalloc(
        logits_buf_, sizeof(T) * batchxbeam * vocab_size_, false));
    cum_log_probs_   = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_    = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    h_finished_buf_  = (bool*)realloc(h_finished_buf_, sizeof(bool) * batchxbeam);

    key_cache_ = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * (2 * self_cache_size + 2 * mem_cache_size), false));
    value_cache_     = key_cache_ + self_cache_size;
    key_mem_cache_   = value_cache_ + self_cache_size;
    value_mem_cache_ = key_mem_cache_ + mem_cache_size;
    attention_mask_  = (T*)(allocator_->reMalloc(attention_mask_, sizeof(T) * (batchxbeam * max_seq_len * max_seq_len), false));

    if (beam_width > 1) {
        cache_indirections_[0] = (int*)(allocator_->reMalloc(
            cache_indirections_[0], sizeof(int) * batchxbeam * (max_seq_len + 1) * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * (max_seq_len + 1);
    }
    tiled_encoder_output_ = (T*)(allocator_->reMalloc(
        tiled_encoder_output_, sizeof(T) * batchxbeam * max_mem_seq_len * encoder_d_model, false));
    tiled_encoder_sequence_length_ =
        (int*)(allocator_->reMalloc(tiled_encoder_sequence_length_, sizeof(int) * batchxbeam, false));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_   = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    output_ids_buf_ =
        (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    parent_ids_buf_ =
        (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    output_ids_transpose_buf_ =
        (int*)(allocator_->reMalloc(output_ids_transpose_buf_, sizeof(int) * batchxbeam * (max_seq_len + 1), false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * (max_seq_len + 1), false));

    if (using_beam_hyps) {
        // Let beam_hyps_ can record at most 2*beam_width because we
        // may find beam_width finished candidates during generation,
        // and may compare them with unfinifhsed another beam_width candidates
        // during finalization.
        beam_hyps_.output_ids_tgt = (int*)allocator_->reMalloc(
            beam_hyps_.output_ids_tgt, sizeof(int) * batchxbeam * 2 * (max_seq_len + 1), true);
        beam_hyps_.sequence_lengths_tgt = (int*)allocator_->reMalloc(
            beam_hyps_.sequence_lengths_tgt, sizeof(int) * batchxbeam * 2, true);
        beam_hyps_.cum_log_probs =
            (float*)allocator_->reMalloc(beam_hyps_.cum_log_probs, sizeof(float) * batchxbeam * 2, true);
        beam_hyps_.normed_scores =
            (float*)allocator_->reMalloc(beam_hyps_.normed_scores, sizeof(float) * batchxbeam * 2, true);
        beam_hyps_.log_probs = (float*)allocator_->reMalloc(
            beam_hyps_.log_probs, sizeof(float) * batchxbeam * 2 * (max_seq_len + 1), true);
        beam_hyps_.min_normed_scores =
            (float*)allocator_->reMalloc(beam_hyps_.min_normed_scores, sizeof(float) * batch_size, true);
        beam_hyps_.num_beams = (int*)allocator_->reMalloc(beam_hyps_.num_beams, sizeof(int) * batch_size, true);
        beam_hyps_.is_done   = (bool*)allocator_->reMalloc(beam_hyps_.is_done, sizeof(bool) * batch_size, true);
    }

    cudaMemsetAsync(attention_mask_, (T)(1.0f), sizeof(T) * (batchxbeam * max_seq_len * max_seq_len), stream_);
    is_allocate_buffer_ = true;
}

template<typename T>
void DebertaDecoding<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        if (deberta_emb_buf_ != nullptr){
            allocator_->free((void**)(&deberta_emb_buf_));
        }
        allocator_->free((void**)(&decoding_sequence_lengths_));
        allocator_->free((void**)(&decoder_input_buf_));
        allocator_->free((void**)(&decoder_output_buf_));
        allocator_->free((void**)(&logits_buf_iter));
        allocator_->free((void**)(&logits_buf_));
        allocator_->free((void**)(&cum_log_probs_));
        allocator_->free((void**)(&finished_buf_));
        free(h_finished_buf_);

        allocator_->free((void**)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void**)(&cache_indirections_)[0]);
        }

        allocator_->free((void**)(&tiled_encoder_output_));
        allocator_->free((void**)(&tiled_encoder_sequence_length_));

        allocator_->free((void**)(&start_ids_buf_));
        allocator_->free((void**)(&end_ids_buf_));

        allocator_->free((void**)(&output_ids_buf_));
        allocator_->free((void**)(&parent_ids_buf_));
        allocator_->free((void**)(&output_ids_transpose_buf_));
        allocator_->free((void**)(&output_log_probs_buf_));

        if (using_beam_hyps) {
            allocator_->free((void**)(&beam_hyps_.output_ids_tgt));
            allocator_->free((void**)(&beam_hyps_.sequence_lengths_tgt));
            allocator_->free((void**)(&beam_hyps_.cum_log_probs));
            allocator_->free((void**)(&beam_hyps_.normed_scores));
            allocator_->free((void**)(&beam_hyps_.log_probs));
            allocator_->free((void**)(&beam_hyps_.min_normed_scores));
            allocator_->free((void**)(&beam_hyps_.num_beams));
            allocator_->free((void**)(&beam_hyps_.is_done));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
DebertaDecoding<T>::DebertaDecoding(size_t                              max_batch_size,
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
                                    ActivationType                      activation_type):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    max_relative_positions_(max_relative_positions),
    relative_position_buckets_(relative_position_buckets),
    inter_size_(inter_size),
    d_model_(d_model),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    q_scaling_(q_scaling),
    start_id_(start_id),
    end_id_(end_id),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    hidden_units_(head_num_ * size_per_head_),
    top_k_(top_k),
    top_p_(top_p),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty),
    activation_type_(activation_type)
{
    vocab_size_padded_ = ceil(vocab_size_ / 8.f) * 8;
    initialize();
}

template<typename T>
DebertaDecoding<T>::DebertaDecoding(DebertaDecoding<T> const& decoding):
    BaseLayer(decoding),
    head_num_(decoding.head_num_),
    size_per_head_(decoding.size_per_head_),
    max_relative_positions_(decoding.max_relative_positions_),
    relative_position_buckets_(decoding.relative_position_buckets_),
    inter_size_(decoding.inter_size_),
    d_model_(decoding.d_model_),
    num_layer_(decoding.num_layer_),
    vocab_size_(decoding.vocab_size_),
    q_scaling_(decoding.q_scaling_),
    start_id_(decoding.start_id_),
    end_id_(decoding.end_id_),
    beam_search_diversity_rate_(decoding.beam_search_diversity_rate_),
    hidden_units_(decoding.hidden_units_),
    top_k_(decoding.top_k_),
    top_p_(decoding.top_p_),
    temperature_(decoding.temperature_),
    len_penalty_(decoding.len_penalty_),
    repetition_penalty_(decoding.repetition_penalty_),
    activation_type_(decoding.activation_type_)
{
    initialize();
}

template<typename T>
DebertaDecoding<T>::~DebertaDecoding()
{
    delete decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

template<typename T>
void DebertaDecoding<T>::registerCallback(callback_sig* fn, void* ctx)
{
    token_generated_cb_  = fn;
    token_generated_ctx_ = ctx;
}

template<typename T>
void DebertaDecoding<T>::unRegisterCallback()
{
    token_generated_cb_  = nullptr;
    token_generated_ctx_ = nullptr;
}

template<typename T>
void DebertaDecoding<T>::forward(std::vector<Tensor>*       output_tensors,
                                 const std::vector<Tensor>* input_tensors,
                                 const TensorMap*           pos_query_cache,
                                 const TensorMap*           pos_key_cache,
                                 const ZcodeDecodingWeight<T>* decoding_weights)
{
    // input_tensors:
    //      input_ids [batch_size, seqlen]
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size]
    //      start_id [batch_size]

    // output_tensors:
    //      output_ids [batch_size, beam, max_seq_len]
    //      output_sequence_length [batch_size, beam], record the number of generated token, except the start token

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the output_sequence_length would be k - 1 before
    // complete this step.

    TensorMap input_tensors_map = TensorMap({
        {"input_ids", input_tensors->at(0)},
        {"encoder_output", input_tensors->at(1)},
        {"encoder_sequence_length", input_tensors->at(2)},
        {"start_id", input_tensors->at(3)}
    });

    TensorMap output_tensors_map = TensorMap({
        {"output_ids", output_tensors->at(0)},
        {"output_sequence_length", output_tensors->at(1)}
    });
    
    forward(&output_tensors_map, &input_tensors_map, pos_query_cache, pos_key_cache, decoding_weights);
}

template<typename T>
void DebertaDecoding<T>::setOutputTensors(TensorMap* output_tensors, TensorMap const* input_tensors)
{
    auto const batch_size       = output_tensors->at("output_ids").shape[0];
    auto const beam_width       = output_tensors->at("output_ids").shape[1];
    auto const batchxbeam       = batch_size * beam_width;
    auto const sequence_lengths = output_tensors->at("output_sequence_length").getPtr<int>();
    auto const max_seq_len      = output_tensors->at("output_ids").shape[2];

    if (beam_width > 1) {
        if (using_beam_hyps) {
            beam_hyps_.sequence_lengths_src = sequence_lengths;
            beam_hyps_.parent_ids_src       = parent_ids_buf_;
            beam_hyps_.output_ids_src       = output_ids_buf_;
            beam_hyps_.log_probs_src        = output_log_probs_buf_;
            beam_hyps_.max_seq_len          = max_seq_len;
            beam_hyps_.length_penalty =
                input_tensors->isExist("len_penalty") ? input_tensors->at("len_penalty").getVal<float>() : 0.0f;

            invokeInsertUnfinishedPath(beam_hyps_, finished_buf_, cum_log_probs_, batch_size, beam_width, stream_);
            sync_check_cuda_error();

            invokeFinalize(output_tensors->getPtr<int>("output_ids"),
                           output_tensors->getPtr<int>("output_sequence_length"),
                           output_tensors->getPtr<float>("cum_log_probs", nullptr),
                           output_tensors->getPtr<float>("output_log_probs", nullptr),
                           beam_hyps_.output_ids_tgt,
                           beam_hyps_.sequence_lengths_tgt,
                           beam_hyps_.normed_scores,
                           beam_hyps_.cum_log_probs,
                           beam_hyps_.log_probs,
                           beam_hyps_.num_beams,
                           beam_width,
                           max_seq_len,
                           batch_size,
                           stream_);
            sync_check_cuda_error();
        }
        else {
            // For beam search, do gather_tree
            invokeGatherTree(output_ids_transpose_buf_,
                             output_tensors->at("output_sequence_length").getPtr<int>(),
                             max_seq_len,
                             batch_size,
                             beam_width,
                             output_ids_buf_ + batchxbeam,
                             parent_ids_buf_ + batchxbeam,
                             end_ids_buf_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  output_ids_transpose_buf_,
                                  max_seq_len,
                                  batchxbeam,
                                  1,
                                  stream_);
        }
    }
    else {
        // For sampling, only transpose the results to output_tensor
        invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                              output_ids_buf_ + batchxbeam,
                              max_seq_len,
                              batchxbeam,
                              1,
                              stream_);
    }

    // Return the cumulative log probability and log probability if requested.
    if (beam_width == 1 || !using_beam_hyps) {
        if (output_tensors->isExist("output_log_probs")) {
            invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                                  output_log_probs_buf_,
                                  max_seq_len,
                                  batchxbeam,
                                  1,
                                  stream_);
        }
        if (output_tensors->isExist("cum_log_probs")) {
            Tensor cum_log_probs = output_tensors->at("cum_log_probs");
            FT_CHECK_WITH_INFO(cum_log_probs.size() == batchxbeam,
                               "The shape of cum_log_probs does not match with batch_size x beam_width.");
            cudaD2Dcpy(cum_log_probs.getPtr<float>(), cum_log_probs_, batchxbeam);
        }
    }

    if (output_tensors->isExist("is_finished")) {
        cudaD2Dcpy(
            output_tensors->at("is_finished").getPtr<bool>(), finished_buf_, output_tensors->at("is_finished").size());
    }
}

template<typename T>
void DebertaDecoding<T>::forward(TensorMap*                      output_tensors, 
                                 TensorMap*                      input_tensors, 
                                 const TensorMap*                pos_query_cache,
                                 const TensorMap*                pos_key_cache,
                                 const ZcodeDecodingWeight<T>*   decoding_weights)
{
    // input_tensors:
    //      input_ids [batch_size, seqlen]
    //      encoder_output [batch_size, mem_max_seq_len, memory_hidden_dimension]
    //      encoder_sequence_length [batch_size]
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [batch_size, 2, stop_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      presence_penalty [1] or [batch_size] on cpu, optional, float.
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional

    // output_tensors:
    //      output_ids [batch_size, beam, max_seq_len]
    //      output_sequence_length [batch_size, beam], record the number of generated token, except the start token
    //      output_log_probs [batch_size, beam, max_seq_len], optional, must be float*.
    //      cum_log_probs [batch_size, beam], optional, must be float*.

    // Step is from 1 ~ max_seq_len,
    // When step = k,  we put output ids and caches at step k, and the output_sequence_length would be k - 1 before
    // complete this step.

    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 2);
    FT_CHECK(output_tensors->size() >= 2);
    FT_CHECK(input_tensors->at("encoder_output").shape[2] == d_model_);
    FT_CHECK(input_tensors->at("encoder_output").shape.size() == 3);

    const size_t decoder_seq_len = input_tensors->at("input_ids").shape[1];
    const size_t batch_size      = output_tensors->at("output_ids").shape[0];
    const size_t beam_width      = output_tensors->at("output_ids").shape[1];
    const size_t batchxbeam      = batch_size * beam_width;
    const size_t max_seq_len     = output_tensors->at("output_ids").shape[2];
    const size_t mem_max_seq_len = input_tensors->at("encoder_output").shape[1];
    allocateBuffer(batch_size, beam_width, decoder_seq_len, max_seq_len, mem_max_seq_len, input_tensors->at("encoder_output").shape[2]);
    
    const int      max_input_length = 1;
    const DataType data_type        = getTensorType<T>();
    int*           sequence_lengths = output_tensors->at("output_sequence_length").getPtr<int>();

    {
        dynamic_decode_layer_->setup(batch_size, beam_width, input_tensors);
        handleOptArg(input_tensors, "start_id", start_ids_buf_, start_id_, batch_size);
        handleOptArg(input_tensors, "end_id", end_ids_buf_, end_id_, batch_size);
        deviceFill(decoding_sequence_lengths_, batchxbeam, (int) 1);
    }
    
    cudaMemsetAsync(output_tensors->at("output_ids").getPtr<int>(), 0, output_tensors->at("output_ids").sizeBytes(), stream_);
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batchxbeam * (max_seq_len + 1), stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batchxbeam * (max_seq_len + 1), stream_);
    
    if (beam_width > 1) {
        cudaMemsetAsync(cache_indirections_[0], 0, 2 * sizeof(int) * batchxbeam * (max_seq_len + 1), stream_);
        invokeTileEncoderResults(tiled_encoder_output_,
                                 tiled_encoder_sequence_length_,
                                 input_tensors->at("encoder_output").getPtr<T>(),
                                 input_tensors->at("encoder_sequence_length").getPtr<const int>(),
                                 batch_size,
                                 beam_width,
                                 mem_max_seq_len,
                                 d_model_,
                                 stream_);
        sync_check_cuda_error();
        encoder_output_ptr_          = tiled_encoder_output_;
        encoder_sequence_length_ptr_ = tiled_encoder_sequence_length_;
    }
    else {
        encoder_output_ptr_          = input_tensors->at("encoder_output").getPtr<const T>();
        encoder_sequence_length_ptr_ = input_tensors->at("encoder_sequence_length").getPtr<const int>();
    }

    invokeDecodingInitialize(finished_buf_,
                             sequence_lengths,
                             output_ids_buf_,
                             cum_log_probs_,
                             start_ids_buf_,
                             batch_size,
                             beam_width,
                             max_input_length - 1,
                             stream_);
    sync_check_cuda_error();

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_ = decoding_weights->word_embedding_table;
    }
    else {
        invokePaddingEmbeddingKernel(padded_embedding_kernel_,
                                     decoding_weights->word_embedding_table,
                                     d_model_,
                                     vocab_size_,
                                     vocab_size_padded_,
                                     stream_);
        sync_check_cuda_error();
    }

    const std::vector<size_t> self_k_cache_shape = {num_layer_, batchxbeam, head_num_, (size_t)(max_seq_len + 1), size_per_head_};
    const std::vector<size_t> self_v_cache_shape = {num_layer_, batchxbeam, head_num_, (size_t)(max_seq_len + 1), size_per_head_};
    const std::vector<size_t> mem_cache_shape    = {num_layer_, batchxbeam, mem_max_seq_len, head_num_ * size_per_head_};
    const size_t local_batch_size = batch_size;
    const uint ite = 0;

    int step_tmp;
    for (int step = max_input_length; step <= (int)max_seq_len; step++) {
        FT_LOG_DEBUG("%s::step: %d", __PRETTY_FUNCTION__, step);
        const int src_indir_idx = beam_width > 1 ? (step - 1) & 0x1 : 0;
        const int tgt_indir_idx = 1 - src_indir_idx;
        
        // Do Step 0: Process Decoder inputs and build KV Cache
        if (step == 1){
            // Word embedding layer [batchxbeam, seq_len] --> [batchxbeam, seq_len, hidden_size]
            invokeInputIdsEmbeddingLookupPosEncoding(
                deberta_emb_buf_,
                nullptr,
                decoding_weights->word_embedding_table,
                (T*)nullptr,  // word embedding only, position embedding was replaced by relative embedding design in
                                // DeBERTa
                pPromptTuningParam<T>{},
                input_tensors->at("input_ids").getPtr<int>(),
                1,
                decoder_seq_len,
                decoder_seq_len,
                batchxbeam,
                hidden_units_,
                stream_);
            sync_check_cuda_error();

            invokeGeneralLayerNorm(deberta_emb_buf_,
                                deberta_emb_buf_,
                                decoding_weights->word_embedding_layernorm_weights.gamma,
                                decoding_weights->word_embedding_layernorm_weights.beta,
                                layernorm_eps_,
                                batchxbeam,
                                hidden_units_,
                                (float*)nullptr,
                                0,
                                stream_);
            sync_check_cuda_error();

            step_tmp = 0;
            TensorMap decoder_input_tensors_step0({
                {"decoder_input", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam, decoder_seq_len, d_model_}, deberta_emb_buf_}},
                {"attention_mask", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam, 1, decoder_seq_len, decoder_seq_len}, attention_mask_}},
                {"current_cache_seq_len", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &step_tmp}},
                {"encoder_output", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam,
                        input_tensors->at("encoder_output").shape[1],
                        input_tensors->at("encoder_output").shape[2]}, encoder_output_ptr_}},
                {"encoder_sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batchxbeam}, encoder_sequence_length_ptr_}},
                {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, std::vector<size_t>{batchxbeam}, finished_buf_}},
                {"step", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &step_tmp}},
                {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &ite}},
                {"cache_indirection", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width, max_seq_len + 1}, 
                        beam_width > 1 ? cache_indirections_[src_indir_idx]: nullptr}}
            });

            TensorMap decoder_output_tensors_step0{
                {"decoder_output", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam, d_model_}, decoder_output_buf_}},
                {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
                {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
                {"key_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, key_mem_cache_}},
                {"value_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, value_mem_cache_}}
            };

            decoder_->forward(&decoder_output_tensors_step0, &decoder_input_tensors_step0, pos_query_cache, pos_key_cache, &decoding_weights->decoder_layer_weights);

            if (is_free_buffer_after_forward_) {
                allocator_->free((void**)(&deberta_emb_buf_));
            }
        }

        // Start Decoding Process
        invokeEmbeddingLookupPosEncodingPadCount(
            decoder_input_buf_,
            decoding_weights->word_embedding_table,
            (T*)nullptr,  // word embedding only, position embedding was replaced by relative embedding design in DeBERTa
            output_ids_buf_,
            (int*)nullptr,
            pPromptTuningParam<T>{},
            batchxbeam,
            hidden_units_,
            (T)1.0f,
            step - 1,
            batchxbeam,
            0, // ite
            1, // seq len
            stream_
        );
        sync_check_cuda_error();

        invokeGeneralLayerNorm(decoder_input_buf_,
                            decoder_input_buf_,
                            decoding_weights->word_embedding_layernorm_weights.gamma,
                            decoding_weights->word_embedding_layernorm_weights.beta,
                            layernorm_eps_,
                            batchxbeam,
                            hidden_units_,
                            (float*)nullptr,
                            0,
                            stream_);
        sync_check_cuda_error();
        
        step_tmp = decoder_seq_len + step - 1;
        TensorMap decoder_input_tensors({
            {"decoder_input", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam, 1, d_model_}, decoder_input_buf_}},
            {"attention_mask", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam, 1, 1, decoder_seq_len}, attention_mask_}},
            {"current_cache_seq_len", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &step_tmp}},
            {"encoder_output", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam,
                    input_tensors->at("encoder_output").shape[1],
                    input_tensors->at("encoder_output").shape[2]}, encoder_output_ptr_}},
            {"encoder_sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batchxbeam}, encoder_sequence_length_ptr_}},
            {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, std::vector<size_t>{batchxbeam}, finished_buf_}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &step}},
            {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &ite}},
            {"cache_indirection", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batch_size, beam_width, max_seq_len + 1}, 
                    beam_width > 1 ? cache_indirections_[src_indir_idx]: nullptr}}
        });

        TensorMap decoder_output_tensors{
            {"decoder_output", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{batchxbeam, d_model_}, decoder_output_buf_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
            {"key_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, key_mem_cache_}},
            {"value_mem_cache", Tensor{MEMORY_GPU, data_type, mem_cache_shape, value_mem_cache_}}
        };

        decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, pos_query_cache, pos_key_cache, &decoding_weights->decoder_layer_weights);
        sync_check_cuda_error();

        // LM Head
        // O = Act(Linear(H, LM_dense_w, LM_dense_b))
        // O = Layernorm(O)
        // Linear(H, WordEmb_w, LM_b)
        
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                d_model_,  // n
                                batchxbeam, // batch
                                d_model_,  // k
                                decoding_weights->lm_head_dense_weights.kernel,
                                d_model_,  // k
                                decoder_output_buf_,
                                d_model_,  // k
                                logits_buf_iter,
                                d_model_ /* n */
                            );

        invokeAddBiasGeluV2(
            logits_buf_iter,
            decoding_weights->lm_head_dense_weights.bias,
            (const int*)nullptr,
            (const T*)nullptr,
            batchxbeam,
            d_model_,
            stream_
        );

        invokeGeneralLayerNorm(logits_buf_iter,
                               logits_buf_iter,
                               decoding_weights->lm_head_layernorm_weights.gamma,
                               decoding_weights->lm_head_layernorm_weights.beta,
                               layernorm_eps_,
                               batchxbeam,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);

        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                vocab_size_padded_,  // n
                                batchxbeam, // batch
                                d_model_,  // k
                                padded_embedding_kernel_ptr_,
                                d_model_,  // k
                                logits_buf_iter,
                                d_model_,  // k
                                decoding_weights->lm_head_bias,
                                logits_buf_,
                                vocab_size_padded_ /* n */
                            );

        // Start Decoding Process
        bool is_initialize_random_table = step == 1;
        TensorMap dynamic_decode_input_tensors({
            {"logits", Tensor{MEMORY_GPU, data_type, std::vector<size_t>{local_batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &step}},
            {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &max_input_length}},
            {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, &ite}},
            {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size}, end_ids_buf_}},
            {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, &local_batch_size}},
            {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, std::vector<size_t>{1}, &is_initialize_random_table}}
        });

        if (cache_indirections_[src_indir_idx] != nullptr) {
            dynamic_decode_input_tensors.insert(
                "src_cache_indirection",
                Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size, beam_width, (max_seq_len + 1)}, cache_indirections_[src_indir_idx]});
        }

        for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
            if (!dynamic_decode_input_tensors.isExist(t->first)) {
                dynamic_decode_input_tensors.insert(*t);
            }
        }

        // common outputs
        TensorMap dynamic_decode_output_tensors({
            {"output_ids",
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(max_seq_len + 1), batch_size, beam_width}, output_ids_buf_}},
            {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, std::vector<size_t>{batchxbeam}, finished_buf_}},
            {"parent_ids",
            Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{(max_seq_len + 1), batch_size, beam_width}, parent_ids_buf_}},
            {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{batchxbeam}, sequence_lengths}}
        });

        if (using_beam_hyps) {
            dynamic_decode_output_tensors.insert("beam_hyps", Tensor{MEMORY_GPU, TYPE_VOID, std::vector<size_t>{1}, &beam_hyps_});
        }

        // cum_log_probs is necessary for beam search, while it is optional for sampling.
        if (beam_width > 1 || output_tensors->isExist("cum_log_probs")) {
            dynamic_decode_output_tensors.insert(
                "cum_log_probs", Tensor{MEMORY_GPU, TYPE_FP32, std::vector<size_t>{batchxbeam}, cum_log_probs_});
        }

        if (output_tensors->getPtr<float>("output_log_probs", nullptr) != nullptr) {
            dynamic_decode_output_tensors.insert(
                "output_log_probs",
                Tensor{
                    MEMORY_GPU, TYPE_FP32, std::vector<size_t>{(max_seq_len + 1), batch_size, beam_width}, output_log_probs_buf_});
        }

        if (cache_indirections_[tgt_indir_idx] != nullptr) {
            dynamic_decode_output_tensors.insert(
                "tgt_cache_indirection",
                Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{local_batch_size, beam_width, (max_seq_len + 1)}, cache_indirections_[tgt_indir_idx]});
        }

        for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
            // Handle exceptions.
            if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                continue;
            }
            dynamic_decode_output_tensors.insert(*t);
        }

        dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);

        cudaD2Hcpy(h_finished_buf_, finished_buf_, batchxbeam);
        uint sum = 0;
        for (uint i = 0; i < batchxbeam; i++) {
            sum += (int)h_finished_buf_[i];
        }
        if (sum == batchxbeam) {
            break;
        }
        else if (step < (int)max_seq_len && token_generated_cb_) {
            setOutputTensors(output_tensors, input_tensors);
            token_generated_cb_(output_tensors, token_generated_ctx_);
        }
    }

    setOutputTensors(output_tensors, input_tensors);

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
}

template class DebertaDecoding<float>;
template class DebertaDecoding<half>;
#ifdef ENABLE_BF16
template class DebertaDecoding<__nv_bfloat16>;
#endif

}  // namespace fastertransformer