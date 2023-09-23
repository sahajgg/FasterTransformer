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

#include "src/fastertransformer/models/zcodepp/ZppEncoderLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct ZppEncoderWeight {

    ZppEncoderWeight() = default;
    ZppEncoderWeight(const size_t hidden_units,
                  const size_t inter_size,
                  const size_t vocab_size,
                  const size_t num_layer);
    ~ZppEncoderWeight();
    ZppEncoderWeight(const ZppEncoderWeight& other);
    ZppEncoderWeight&                     operator=(const ZppEncoderWeight& other);
    std::vector<ZppEncoderLayerWeight<T>> zpp_encoder_layer_weights;
    const T*                           word_embedding_table = nullptr;
    LayerNormWeight<T>                 word_embedding_layernorm_weights;

    void loadModel(std::string dir_path);

private:
    void setWeightPtr();

    size_t hidden_units_;
    size_t inter_size_;
    size_t vocab_size_;
    size_t num_layer_;
    bool   is_maintain_buffer = false;

    // 3: [1] word embedding weight [2] word-LN weight [3] word-LN bias
    const static int weights_num_ = 3;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
