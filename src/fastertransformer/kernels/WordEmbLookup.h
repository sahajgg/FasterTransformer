/*
 * Copyright (c) 2019-2023, sahajgg.  All rights reserved.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/memory_utils.h"

template<typename T>
void invokeInputIdsWordEmbeddingLookup(T*                    from_tensor,
                                        const T*              embedding_table,
                                        const int*            input_ids,
                                        const int             length,
                                        const int             max_length,
                                        const int             batch_size,
                                        const int             hidden_units,
                                        cudaStream_t          stream);