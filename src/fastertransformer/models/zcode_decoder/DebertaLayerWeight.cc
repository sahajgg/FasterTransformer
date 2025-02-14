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

#include "DebertaLayerWeight.h"

namespace fastertransformer {

template<typename T>
ZcodeDecoderLayerWeight<T>::ZcodeDecoderLayerWeight(const size_t hidden_units,
                                          const size_t inter_size,
                                          const size_t tensor_para_size,
                                          const size_t tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::string name;

    // Note: TensorFlow/PyTorch interface usage is unclear at this time, so weight loading are based on named bin for
    // now
    name = "attention.self.query_proj.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.self.query_proj.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.self.key_proj.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.self.key_proj.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.self.value_proj.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.self.value_proj.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "attention.output.dense.bias.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    name = "attention.output.LayerNorm.weight.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    name = "attention.output.LayerNorm.bias.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});

    name = "cross_attention.self.query_proj.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.self.query_proj.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.self.key_proj.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.self.key_proj.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.self.value_proj.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.self.value_proj.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, hidden_units_ / tensor_para_size_}, nullptr)});
    name = "cross_attention.output.dense.bias.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    name = "cross_attention.output.LayerNorm.weight.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    name = "cross_attention.output.LayerNorm.bias.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    
    name = "intermediate.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_, inter_size_ / tensor_para_size_}, nullptr)});
    name = "intermediate.dense.bias." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {inter_size_ / tensor_para_size_}, nullptr)});
    name = "output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {inter_size_ / tensor_para_size_, hidden_units_}, nullptr)});
    name = "output.dense.bias.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    name = "output.LayerNorm.weight.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});
    name = "output.LayerNorm.bias.bin";
    weights_ptr_.insert({name, FtWeight<T>(name, {hidden_units_}, nullptr)});

    for (auto it = weights_ptr_.begin(); it != weights_ptr_.end(); ++it) {
        deviceMalloc(&it->second.ptr_, it->second.size_);
    }
    setWeightPtr();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
ZcodeDecoderLayerWeight<T>::~ZcodeDecoderLayerWeight()
{
    if (is_maintain_buffer_ == true) {
        for (auto it = weights_ptr_.begin(); it != weights_ptr_.end(); ++it) {
            deviceFree(it->second.ptr_);
        }
        weights_ptr_.clear();

        attention_weights.query_weight.kernel            = nullptr;
        attention_weights.query_weight.bias              = nullptr;
        attention_weights.key_weight.kernel              = nullptr;
        attention_weights.key_weight.bias                = nullptr;
        attention_weights.value_weight.kernel            = nullptr;
        attention_weights.value_weight.bias              = nullptr;
        attention_weights.attention_output_weight.kernel = nullptr;
        attention_weights.attention_output_weight.bias   = nullptr;
        attn_layernorm_weights.gamma                     = nullptr;
        attn_layernorm_weights.beta                      = nullptr;

        cross_attention_weights.query_weight.kernel            = nullptr;
        cross_attention_weights.query_weight.bias              = nullptr;
        cross_attention_weights.key_weight.kernel              = nullptr;
        cross_attention_weights.key_weight.bias                = nullptr;
        cross_attention_weights.value_weight.kernel            = nullptr;
        cross_attention_weights.value_weight.bias              = nullptr;
        cross_attention_weights.attention_output_weight.kernel = nullptr;
        cross_attention_weights.attention_output_weight.bias   = nullptr;
        cross_attn_layernorm_weights.gamma                     = nullptr;
        cross_attn_layernorm_weights.beta                      = nullptr;

        ffn_weights.intermediate_weight.kernel           = nullptr;
        ffn_weights.intermediate_weight.bias             = nullptr;
        ffn_weights.output_weight.kernel                 = nullptr;
        ffn_weights.output_weight.bias                   = nullptr;
        ffn_layernorm_weights.gamma                      = nullptr;
        ffn_layernorm_weights.beta                       = nullptr;
        is_maintain_buffer_                              = false;
    }
}

template<typename T>
ZcodeDecoderLayerWeight<T>::ZcodeDecoderLayerWeight(const ZcodeDecoderLayerWeight& other):
    ZcodeDecoderLayerWeight(other.hidden_units_, other.inter_size_, other.tensor_para_size_, other.tensor_para_rank_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto it = other.weights_ptr_.begin(); it != other.weights_ptr_.end(); ++it) {
        cudaD2Dcpy(weights_ptr_.at(it->first).ptr_, it->second.ptr_, it->second.size_);
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
ZcodeDecoderLayerWeight<T>& ZcodeDecoderLayerWeight<T>::operator=(const ZcodeDecoderLayerWeight& other)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    hidden_units_     = other.hidden_units_;
    inter_size_       = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;

    for (auto it = other.weights_ptr_.begin(); it != other.weights_ptr_.end(); ++it) {
        weights_ptr_.insert({it->first, it->second});
        weights_ptr_.at(it->first).ptr_ = nullptr;
        deviceMalloc(&weights_ptr_.at(it->first).ptr_, it->second.size_);
        cudaD2Dcpy(weights_ptr_.at(it->first).ptr_, it->second.ptr_, it->second.size_);
    }
    setWeightPtr();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);

    return *this;
}

template<typename T>
void ZcodeDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto it = weights_ptr_.begin(); it != weights_ptr_.end(); ++it) {
        loadWeightFromBin<T>(it->second.ptr_, it->second.shape_, dir_path + it->first, model_file_type);
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void ZcodeDecoderLayerWeight<T>::setWeightPtr()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    attention_weights.query_weight.kernel =
        weights_ptr_.at("attention.self.query_proj.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.query_weight.bias =
        weights_ptr_.at("attention.self.query_proj.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.key_weight.kernel =
        weights_ptr_.at("attention.self.key_proj.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.key_weight.bias =
        weights_ptr_.at("attention.self.key_proj.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.value_weight.kernel =
        weights_ptr_.at("attention.self.value_proj.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.value_weight.bias =
        weights_ptr_.at("attention.self.value_proj.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.attention_output_weight.kernel =
        weights_ptr_.at("attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    attention_weights.attention_output_weight.bias = weights_ptr_.at("attention.output.dense.bias.bin").ptr_;
    attn_layernorm_weights.gamma                   = weights_ptr_.at("attention.output.LayerNorm.weight.bin").ptr_;
    attn_layernorm_weights.beta                    = weights_ptr_.at("attention.output.LayerNorm.bias.bin").ptr_;
    
    cross_attention_weights.query_weight.kernel =
        weights_ptr_.at("cross_attention.self.query_proj.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.query_weight.bias =
        weights_ptr_.at("cross_attention.self.query_proj.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.key_weight.kernel =
        weights_ptr_.at("cross_attention.self.key_proj.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.key_weight.bias =
        weights_ptr_.at("cross_attention.self.key_proj.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.value_weight.kernel =
        weights_ptr_.at("cross_attention.self.value_proj.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.value_weight.bias =
        weights_ptr_.at("cross_attention.self.value_proj.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.attention_output_weight.kernel =
        weights_ptr_.at("cross_attention.output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    cross_attention_weights.attention_output_weight.bias = weights_ptr_.at("cross_attention.output.dense.bias.bin").ptr_;
    cross_attn_layernorm_weights.gamma                   = weights_ptr_.at("cross_attention.output.LayerNorm.weight.bin").ptr_;
    cross_attn_layernorm_weights.beta                    = weights_ptr_.at("cross_attention.output.LayerNorm.bias.bin").ptr_;
    
    ffn_weights.intermediate_weight.kernel =
        weights_ptr_.at("intermediate.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    ffn_weights.intermediate_weight.bias =
        weights_ptr_.at("intermediate.dense.bias." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    ffn_weights.output_weight.kernel =
        weights_ptr_.at("output.dense.weight." + std::to_string(tensor_para_rank_) + ".bin").ptr_;
    ffn_weights.output_weight.bias = weights_ptr_.at("output.dense.bias.bin").ptr_;
    ffn_layernorm_weights.gamma    = weights_ptr_.at("output.LayerNorm.weight.bin").ptr_;
    ffn_layernorm_weights.beta     = weights_ptr_.at("output.LayerNorm.bias.bin").ptr_;

    is_maintain_buffer_ = true;
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template struct ZcodeDecoderLayerWeight<float>;
template struct ZcodeDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ZcodeDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
