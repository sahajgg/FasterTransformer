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

#include "ZppEncoderWeight.h"

namespace fastertransformer {

template<typename T>
ZppEncoderWeight<T>::ZppEncoderWeight(const size_t hidden_units,
                                const size_t inter_size,
                                const size_t vocab_size,
                                const size_t num_layer):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // [1] word embedding weight [2] word-LN weight [3] word-LN bias
    weights_size[0] = vocab_size_ * hidden_units_;
    weights_size[1] = hidden_units_;
    weights_size[2] = hidden_units_;

    for (int i = 0; i < weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }

    setWeightPtr();
    zpp_encoder_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        zpp_encoder_layer_weights.push_back(
            ZppEncoderLayerWeight<T>(hidden_units_, inter_size_));
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
ZppEncoderWeight<T>::~ZppEncoderWeight()
{
    if (is_maintain_buffer == true) {
        zpp_encoder_layer_weights.clear();
        for (int i = 0; i < weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        word_embedding_table                       = nullptr;
        word_embedding_layernorm_weights.gamma     = nullptr;
        word_embedding_layernorm_weights.beta      = nullptr;
        is_maintain_buffer = false;
    }
}

template<typename T>
void ZppEncoderWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FtCudaDataType model_file_type = FtCudaDataType::FP16;

    loadWeightFromBin<T>(weights_ptr[0], {(size_t)weights_size[0]}, dir_path + "/model.e.embeddings.word_embeddings.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {(size_t)weights_size[1]}, dir_path + "/model.e.embeddings.LayerNorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[2], {(size_t)weights_size[2]}, dir_path + "/model.e.embeddings.LayerNorm.bias.bin", model_file_type);

    for (uint l = 0; l < num_layer_; l++) {
        zpp_encoder_layer_weights[l].loadModel(dir_path + "/model.e.encoder.layer." + std::to_string(l) + ".", model_file_type);
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__, " stop");
}

template<typename T>
void ZppEncoderWeight<T>::setWeightPtr()
{
    word_embedding_table                   = weights_ptr[0];
    word_embedding_layernorm_weights.gamma = weights_ptr[1];
    word_embedding_layernorm_weights.beta  = weights_ptr[2];
    
    is_maintain_buffer = true;
}

template struct ZppEncoderWeight<half>;

}  // namespace fastertransformer
