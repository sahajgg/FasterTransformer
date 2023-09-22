/*
 * Copyright (c) 2020-2023, sahajgg.  All rights reserved.
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

#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "src/fastertransformer/kernels/WordEmbLookup.h"
#include "src/fastertransformer/utils/memory_utils.h"

template<typename T>
__global__ void WordEmbLookup(T*                    from_tensor,
                                const T*              embedding_table,
                                const int*            input_ids,
                                const int             length,
                                const int             max_length,
                                const int             batch_size,
                                const int64_t         hidden_units)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * length * hidden_units;
         index += blockDim.x * gridDim.x) {

        // embedding lookup from word ids [batch, length] (part of [batch, max_length]) and [vocab, hidden] to generate
        // embedding [batch, length, hidden]
        const int word_index      = index / hidden_units;
        const int word_index_row  = word_index / length;  // batch_id
        const int word_index_col  = word_index % length;
        const int real_word_index = word_index_row * max_length + word_index_col;
        const int col_index       = index % hidden_units;
        const int input_id        = input_ids == nullptr ? real_word_index : input_ids[real_word_index];
        
        from_tensor[index] = embedding_table[input_id * hidden_units + col_index];
    }
}

template<typename T>
void invokeInputIdsWordEmbeddingLookup(T*                    from_tensor,
                                        const T*              embedding_table,  // can also be inputs_embeds
                                        const int*            input_ids,
                                        const int             length,
                                        const int             max_length,
                                        const int             batch_size,
                                        const int             hidden_units,
                                        cudaStream_t          stream)
{
    dim3       grid(min(batch_size * length, 65536));
    dim3       block(min(hidden_units, 512));
    WordEmbLookup<T><<<grid, block, 0, stream>>>(from_tensor, embedding_table, input_ids, length, max_length, batch_size, hidden_units);
}

template void invokeInputIdsWordEmbeddingLookup(half*                    from_tensor,
                                                const half*              embedding_table,
                                                const int*               input_ids,
                                                const int                length,
                                                const int                max_length,
                                                const int                batch_size,
                                                const int                hidden_units,
                                                cudaStream_t             stream);