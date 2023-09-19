# Copyright (c) 2023, sahajgg.  All rights reserved.
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

add_library(ZppEncoderLayerWeight STATIC ZppEncoderLayerWeight.cc)
set_property(TARGET ZppEncoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZppEncoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZppEncoderLayerWeight PUBLIC memory_utils)

add_library(ZppEncoderWeight STATIC ZppEncoderWeight.cc)
set_property(TARGET ZppEncoderWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZppEncoderWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZppEncoderWeight PUBLIC ZppEncoderLayerWeight)

add_library(ZppEncoder STATIC ZppEncoder.cc)
set_property(TARGET ZppEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZppEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZppEncoder PUBLIC -lcudart bert_preprocess_kernels gpt_kernels WordEmbLookup layernorm_kernels cublasMMWrapper ZppEncoderAttentionLayer GeluFfnLayer add_residual_kernels ZppEncoderWeight nccl_utils custom_ar_comm tensor cuda_utils logger)
