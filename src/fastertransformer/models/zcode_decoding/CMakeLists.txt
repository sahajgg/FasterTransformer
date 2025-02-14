# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(ZcodeDecodingWeight STATIC DebertaWeight.cc)
set_property(TARGET ZcodeDecodingWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZcodeDecodingWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZcodeDecodingWeight PUBLIC ZcodeDecoderLayerWeight)

add_library(DebertaDecoding STATIC DebertaDecoding.cc)
set_property(TARGET DebertaDecoding PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET DebertaDecoding PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(DebertaDecoding PUBLIC -lcudart cublasMMWrapper ZcodeDecoder ZcodeDecodingWeight bert_preprocess_kernels
                                        decoding_kernels DynamicDecodeLayer BaseBeamSearchLayer 
                                        beam_search_topk_kernels gpt_kernels layernorm_kernels activation_kernels tensor cuda_utils logger)