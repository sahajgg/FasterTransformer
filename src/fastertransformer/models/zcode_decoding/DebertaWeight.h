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

#pragma once

#include "src/fastertransformer/models/zcode_decoder/DebertaLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct ZcodeDecodingWeight {

    ZcodeDecodingWeight() = default;
    ZcodeDecodingWeight(const size_t hidden_units,
                  const size_t inter_size,
                  const size_t max_relative_positions,
                  const size_t relative_position_buckets,
                  const size_t vocab_size,
                  const size_t num_layer);

    ~ZcodeDecodingWeight();
    ZcodeDecodingWeight(const ZcodeDecodingWeight& other);
    ZcodeDecodingWeight&                     operator=(const ZcodeDecodingWeight& other);
    std::vector<ZcodeDecoderLayerWeight<T>>    decoder_layer_weights;
    DenseWeight<T>                             lm_head_dense_weights;
    LayerNormWeight<T>                         lm_head_layernorm_weights;
    const T*                                   word_embedding_table = nullptr;
    LayerNormWeight<T>                         word_embedding_layernorm_weights;
    const T*                                   lm_head_bias = nullptr;
;
    void loadModel(std::string dir_path);

private:
    void setWeightPtr();

    size_t hidden_units_;
    size_t inter_size_;
    size_t max_relative_positions_;
    size_t relative_position_buckets_;
    size_t vocab_size_;
    size_t num_layer_;
    bool   is_maintain_buffer = false;

    // 8: [1] word embedding weight [2] word-LN weight [3] word-LN bias [4-5] lm head weight [6-7] lm head LN weight [8] lm head bias
    const static int weights_num_ = 8;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
