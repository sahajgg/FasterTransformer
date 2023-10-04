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

#include "src/fastertransformer/layers/attention_layers/DecoderCrossAttentionLayerOpt.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFTCrossAttentionKernel {
public:
    virtual ~IFTCrossAttentionKernel() {}
    virtual th::Tensor forward(
                        th::Tensor&               query,
                        th::Tensor&               key_cache,
                        th::Tensor&               value_cache,
                        th::Tensor&               encoder_sequence_length,
                        th::Tensor&               step,
                        th::Tensor&               finished) = 0;
};

template<typename T>
class FTCrossAttentionKernel: public IFTCrossAttentionKernel {
public:
    FTCrossAttentionKernel(size_t                         head_num,
                        size_t                         head_size,
                        float                          q_scaling,
                        const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _q_scaling(q_scaling),
        _weights(w)
    {
        ft::ftNcclInitialize(tensor_para_, pipeline_para_, 1, 1);
        ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));

        cublas_algo_map_            = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_       = new std::mutex();

        cross_attention_weights.query_weight.kernel = get_ptr<T>(_weights[0]);
        cross_attention_weights.query_weight.bias = get_ptr<T>(_weights[1]);
        cross_attention_weights.key_weight.bias = get_ptr<T>(_weights[2]);
        cross_attention_weights.value_weight.bias = get_ptr<T>(_weights[3]);
        cross_attention_weights.attention_output_weight.kernel = get_ptr<T>(_weights[4]);
        cross_attention_weights.attention_output_weight.bias = get_ptr<T>(_weights[5]);
        cross_attn_layernorm_weights.gamma = get_ptr<T>(_weights[6]);
        cross_attn_layernorm_weights.beta = get_ptr<T>(_weights[7]);
    }

    ~FTCrossAttentionKernel() override
    {
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(_cublasltHandle);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    th::Tensor forward(
                th::Tensor&               query,
                th::Tensor&               key_cache,
                th::Tensor&               value_cache,
                th::Tensor&               encoder_sequence_length,
                th::Tensor&               step,
                th::Tensor&               finished) override
    {
        auto           stream        = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper* cublas_wrapper = new ft::cublasMMWrapper(_cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        cublas_wrapper->setFP16GemmConfig();

        const size_t batch_size      = (size_t)value_cache.size(0);
        const size_t mem_max_seq_len = (size_t)value_cache.size(2);
        const size_t d_model         = (size_t) (_head_num * _head_size);

        ft::DecoderCrossAttentionLayerOpt<T>* crossattentionkernel = new ft::DecoderCrossAttentionLayerOpt<T>(batch_size,
                                                                                                _head_num,
                                                                                                _head_size,
                                                                                                d_model,
                                                                                                _q_scaling,
                                                                                                stream,
                                                                                                cublas_wrapper,
                                                                                                allocator,
                                                                                                false);

        ft::DataType            data_type     = ft::getTensorType<T>();

        ft::TensorMap input_tensors({
            {"input_query", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, d_model}, get_ptr<T>(query)}},
            {"key_cache", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, _head_num, _head_size / (16 / sizeof(T)), mem_max_seq_len, 16 / sizeof(T)}, get_ptr<T>(key_cache)}},
            {"value_cache", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, _head_num, mem_max_seq_len, _head_size}, get_ptr<T>(value_cache)}},
            {"encoder_sequence_length", ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, get_ptr<int>(encoder_sequence_length)}},
            {"step", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, get_ptr<int>(step)}},
            {"finished", ft::Tensor{ft::MEMORY_GPU, ft::TYPE_BOOL, std::vector<size_t>{batch_size}, get_ptr<int>(finished)}}
        });

        auto attention_output = torch::empty({batch_size, 1, d_model}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
        ft::TensorMap output_tensors(
            {{"attention_output", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, d_model}, get_ptr<T>(attention_output)}}}
        );

        try {
            crossattentionkernel->forward(&output_tensors, &input_tensors, &cross_attention_weights, &cross_attn_layernorm_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete crossattentionkernel;
        delete cublas_wrapper;
        delete allocator;
        return attention_output;
    }

private:
    const size_t            _head_num;
    const size_t            _head_size;
    const float             _q_scaling;
    std::vector<th::Tensor> _weights;

    ft::NcclParam           tensor_para_;
    ft::NcclParam           pipeline_para_;
    cublasLtHandle_t        _cublasltHandle;
    std::mutex*          cublas_wrapper_mutex_;
    ft::cublasAlgoMap*   cublas_algo_map_;
    ft::AttentionWeight<T> cross_attention_weights;
    ft::LayerNormWeight<T> cross_attn_layernorm_weights;
};

class FasterTransformerCrossAttentionKernel: public th::jit::CustomClassHolder {
public:
    FasterTransformerCrossAttentionKernel(
                                th::Tensor           cross_q_kernel,
                                th::Tensor           cross_q_bias,
                                th::Tensor           cross_k_bias,
                                th::Tensor           cross_v_bias,
                                th::Tensor           cross_attn_output_kernel,
                                th::Tensor           cross_attn_output_bias,
                                th::Tensor           cross_attn_layernorm_gamma,
                                th::Tensor           cross_attn_layernorm_beta,
                                int64_t              head_num,
                                int64_t              size_per_head,
                                double               q_scaling);

    ~FasterTransformerCrossAttentionKernel();

    th::Tensor forward(
        th::Tensor               query,
        th::Tensor               key_cache,
        th::Tensor               value_cache,
        th::Tensor               encoder_sequence_length,
        th::Tensor               step,
        th::Tensor               finished
    );

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType        _st;
    IFTCrossAttentionKernel*    ftcrossattentionkernel;
    std::vector<th::Tensor>     weights;
};

}  // namespace torch_ext
