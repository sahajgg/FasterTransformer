# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

add_library(ZcodeEncoderLayerWeight STATIC DebertaLayerWeight.cc)
set_property(TARGET ZcodeEncoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZcodeEncoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZcodeEncoderLayerWeight PUBLIC memory_utils)

add_library(ZcodeEncoderWeight STATIC DebertaWeight.cc)
set_property(TARGET ZcodeEncoderWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZcodeEncoderWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZcodeEncoderWeight PUBLIC ZcodeEncoderLayerWeight)

add_library(ZcodeEncoder STATIC Deberta.cc)
set_property(TARGET ZcodeEncoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ZcodeEncoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ZcodeEncoder PUBLIC -lcudart bert_preprocess_kernels gpt_kernels layernorm_kernels cublasMMWrapper ZcodeEncoderDisentangledAttentionLayer FfnLayer add_residual_kernels ZcodeEncoderWeight nccl_utils custom_ar_comm tensor cuda_utils logger)
