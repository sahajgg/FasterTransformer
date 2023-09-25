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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/zcode_decoder/DebertaLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct ZcodeDecoderWeight {

    ZcodeDecoderWeight() = default;
    ZcodeDecoderWeight(const int hidden_units,
                   const int inter_size,
                   const int max_relative_positions,
                   const int relative_position_buckets,
                   const int vocab_size,
                   const int num_layer,
                   const int max_seq_len):
        hidden_units_(hidden_units),
        inter_size_(inter_size),
        max_relative_positions_(max_relative_positions),
        relative_position_buckets_(relative_position_buckets),
        vocab_size_(vocab_size),
        num_layer_(num_layer),
        max_seq_len_(max_seq_len)
    {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);

        mallocWeights();
        setWeightPtr();
        deberta_layer_weights.reserve(num_layer_);
        for (int i = 0; i < num_layer_; i++) {
            deberta_layer_weights.push_back(ZcodeDecoderLayerWeight<T>(hidden_units_, inter_size_));
        }
        FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    }

    ~ZcodeDecoderWeight()
    {
        if (is_maintain_buffer == true) {
            decoder_layer_weights.clear();
            for (int i = 0; i < 7; i++) {
                deviceFree(weights_ptr[i]);
            }

            word_embedding_table                    = nullptr;
            word_embedding_layernorm_weights.beta   = nullptr;
            word_embedding_layernorm_weights.gamma  = nullptr;
            post_decoder_embedding.kernel           = nullptr;
            post_decoder_embedding.bias             = nullptr;
            post_decoder_layernorm.beta             = nullptr;
            post_decoder_layernorm.gamma            = nullptr;

            is_maintain_buffer                      = false;
        }
    }

    ZcodeDecoderWeight(const ZcodeDecoderWeight& other):
        hidden_units_(other.hidden_units_),
        inter_size_(other.inter_size_),
        max_relative_positions_(other.max_relative_positions_),
        relative_position_buckets_(other.relative_position_buckets_),
        num_layer_(other.num_layer_),
        vocab_size_(other.vocab_size_),
        max_seq_len_(other.max_seq_len_)
    {
        mallocWeights();
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
        setWeightPtr();

        decoder_layer_weights.clear();
        for (int l = 0; l < num_layer_; l++) {
            decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
        }
    }

    ZcodeDecoderWeight& operator=(const ZcodeDecoderWeight& other)
    {
        hidden_units_     = other.hidden_units_;
        inter_size_       = other.inter_size_;
        max_relative_positions_ = other.max_relative_positions_;
        relative_position_buckets_ = other.relative_position_buckets_;
        num_layer_        = other.num_layer_;
        vocab_size_       = other.vocab_size_;
        max_seq_len_      = other.max_seq_len_;

        mallocWeights();
        cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], vocab_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
        cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
        setWeightPtr();

        decoder_layer_weights.clear();
        for (int l = 0; l < num_layer_; l++) {
            decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
        }
        return *this;
    }

    void mallocWeights()
    {

        deviceMalloc(&weights_ptr[0], vocab_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[1], hidden_units_);
        deviceMalloc(&weights_ptr[2], hidden_units_);
        deviceMalloc(&weights_ptr[3], hidden_units_ * hidden_units_);
        deviceMalloc(&weights_ptr[4], hidden_units_);
        deviceMalloc(&weights_ptr[5], hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_);
        is_maintain_buffer = true;
    }

    std::vector<ZcodeDecoderLayerWeight<T>> decoder_layer_weights;
    
    const T*                           word_embedding_table = nullptr;
    LayerNormWeight<T>                 word_embedding_layernorm_weights;
    
    DenseWeight<T>                     post_decoder_embedding;
    LayerNormWeight<T>                 post_decoder_layernorm;
    

private:
    void setWeightPtr()
    {
        word_embedding_table                    = weights_ptr[0];
        word_embedding_layernorm_weights.beta   = weights_ptr[1];
        word_embedding_layernorm_weights.gamma  = weights_ptr[2];
        post_decoder_embedding.kernel           = weights_ptr[3];
        post_decoder_embedding.bias             = weights_ptr[4];
        post_decoder_layernorm.beta             = weights_ptr[5];
        post_decoder_layernorm.gamma            = weights_ptr[6];
    }

    int  hidden_units_;
    int  inter_size_;
    int  max_relative_positions_;
    int  relative_position_buckets_;
    int  vocab_size_;
    int  num_layer_;
    int  max_seq_len_;
    bool is_maintain_buffer = false;
    T*   weights_ptr[7];
};

}  // namespace fastertransformer
