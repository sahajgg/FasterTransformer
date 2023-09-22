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

#include "src/fastertransformer/models/zcodepp/ZppEncoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFZppEncoderModel {
public:
    virtual ~IFZppEncoderModel() {}
    virtual void forward(long unsigned int batch_size,
                        long unsigned int seq_len,
                        th::Tensor& input,
                        th::Tensor& sequence_lengths,
                        std::vector<th::Tensor>& pos_query_cache,
                        std::vector<th::Tensor>& pos_key_cache,
                        th::Tensor& output) = 0;
};



template<typename T>
class FTZppEncoderModel: public IFZppEncoderModel {
public:
    FTZppEncoderModel(size_t                         head_num,
                    size_t                         head_size,
                    size_t                         inter_size,
                    size_t                         layer_num,
                    size_t                         position_buckets,
                    float                          q_scaling,
                    const std::vector<th::Tensor>& w):
        _head_num(head_num),
        _head_size(head_size),
        _inter_size(inter_size),
        _layer_num(layer_num),
        _position_buckets(position_buckets),
        _weights(w),
        _q_scaling(q_scaling)
    {

        const int hidden_dim       = _head_num * _head_size;
        ft::check_cuda_error(cublasLtCreate(&_cublasltHandle));

        cublas_algo_map_            = new ft::cublasAlgoMap("gemm_config.in", "");
        cublas_wrapper_mutex_       = new std::mutex();

        // Layer-level weights
        zppencoder_weights.zpp_encoder_layer_weights.clear();
        zppencoder_weights.zpp_encoder_layer_weights.resize(_layer_num);
        for (int i = 0; i < _layer_num; i++) {
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.query_weight.kernel =
                get_ptr<T>(_weights[0]) + hidden_dim * hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.query_weight.bias =
                get_ptr<T>(_weights[1]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.key_weight.kernel =
                get_ptr<T>(_weights[2]) + hidden_dim * hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.key_weight.bias =
                get_ptr<T>(_weights[3]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.value_weight.kernel =
                get_ptr<T>(_weights[4]) + hidden_dim * hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.value_weight.bias =
                get_ptr<T>(_weights[5]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.attention_output_weight.kernel =
                get_ptr<T>(_weights[6]) + hidden_dim * hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.attention_output_weight.bias =
                get_ptr<T>(_weights[7]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attn_layernorm_weights.gamma =
                get_ptr<T>(_weights[8]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].attn_layernorm_weights.beta =
                get_ptr<T>(_weights[9]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(_weights[10])
                + hidden_dim * _inter_size * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.intermediate_weight.bias =
                get_ptr<T>(_weights[11]) + _inter_size * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.output_weight.kernel =
                get_ptr<T>(_weights[12])
                + _inter_size * hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.output_weight.bias =
                get_ptr<T>(_weights[13]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_layernorm_weights.gamma =
                get_ptr<T>(_weights[14]) + hidden_dim * i;
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_layernorm_weights.beta =
                get_ptr<T>(_weights[15]) + hidden_dim * i;
        }

        // Model-level weights
        zppencoder_weights.word_embedding_table                       = get_ptr<T>(_weights[16]);
        zppencoder_weights.word_embedding_layernorm_weights.gamma     = get_ptr<T>(_weights[17]);
        zppencoder_weights.word_embedding_layernorm_weights.beta      = get_ptr<T>(_weights[18]);
    }

    ~FTZppEncoderModel() override
    {
        cublasLtDestroy(_cublasltHandle);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(long unsigned int batch_size,
                 long unsigned int seq_len,
                 th::Tensor& input,
                 th::Tensor& sequence_lengths,
                 std::vector<th::Tensor>& pos_query_cache,
                 std::vector<th::Tensor>& pos_key_cache,
                 th::Tensor& output) override
    {
        auto           stream        = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t _cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(_cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH>* allocator = new ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper*                  cublas_wrapper =

            new ft::cublasMMWrapper(
                _cublasHandle, _cublasltHandle, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        cublas_wrapper->setFP16GemmConfig();

        ft::ZppEncoder<T>* zppencoder = new ft::ZppEncoder<T>(_head_num,
                                                            _head_size,
                                                            _inter_size,
                                                            _layer_num,
                                                            _position_buckets,
                                                            _q_scaling,
                                                            stream,
                                                            cublas_wrapper,
                                                            allocator,
                                                            true);

        ft::DataType            data_type     = ft::getTensorType<T>();
        std::vector<ft::Tensor> input_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, get_ptr<int>(input)},
            ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, get_ptr<int>(sequence_lengths)}};

        std::vector<ft::Tensor> ft_pos_query_cache = std::vector<ft::Tensor>{};
        std::vector<ft::Tensor> ft_pos_key_cache   = std::vector<ft::Tensor>{};
        for(uint l=0; l < _layer_num; l++)
        {
            ft_pos_query_cache.push_back(ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, _head_num, 2 * _position_buckets, _head_size}, get_ptr<T>(pos_query_cache[l])});
            ft_pos_key_cache.push_back(ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, _head_num, 2 * _position_buckets, _head_size}, get_ptr<T>(pos_key_cache[l])});
        }

        std::vector<ft::Tensor> output_tensors = std::vector<ft::Tensor>{
            ft::Tensor{ft::MEMORY_GPU,
                       data_type,
                       std::vector<size_t>{batch_size, seq_len, (size_t)(_head_num * _head_size)},
                       get_ptr<T>(output)}};

        try {
            zppencoder->forward(
                &output_tensors, 
                &input_tensors,
                &ft_pos_query_cache,
                &ft_pos_key_cache,
                &zppencoder_weights);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
        delete zppencoder;
        delete cublas_wrapper;
        delete allocator;
    }

private:
    const size_t            _head_num;
    const size_t            _head_size;
    const size_t            _inter_size;
    const size_t            _layer_num;
    const size_t            _position_buckets;
    std::vector<th::Tensor> _weights;
    const float             _q_scaling;
    cublasLtHandle_t        _cublasltHandle;
    std::mutex*          cublas_wrapper_mutex_;
    ft::cublasAlgoMap*   cublas_algo_map_;
    ft::ZppEncoderWeight<T> zppencoder_weights;
};

class FasterTransformerZppEncoderModel: public th::jit::CustomClassHolder {
public:
    FasterTransformerZppEncoderModel(th::Tensor q_kernel,
                             th::Tensor q_bias,
                             th::Tensor k_kernel,
                             th::Tensor k_bias,
                             th::Tensor v_kernel,
                             th::Tensor v_bias,
                             th::Tensor attr_output_kernel,
                             th::Tensor attr_output_bias,
                             th::Tensor attr_output_layernorm_gamma,
                             th::Tensor attr_output_layernorm_beta,
                             th::Tensor inter_kernel,
                             th::Tensor inter_bias,
                             th::Tensor output_kernel,
                             th::Tensor output_bias,
                             th::Tensor output_layernorm_gamma,
                             th::Tensor output_layernorm_beta,
                             th::Tensor word_embedding_table,
                             th::Tensor word_embedding_layernorm_gamma,
                             th::Tensor word_embedding_layernorm_beta,
                             int64_t    head_num,
                             int64_t    head_size,
                             int64_t    inter_size,
                             int64_t    layer_num,
                             int64_t    position_buckets,
                             double     q_scaling);

    ~FasterTransformerZppEncoderModel();

    th::Tensor forward(
        th::Tensor input, 
        th::Tensor sequence_lengths,
        std::vector<th::Tensor> pos_query_cache,
        std::vector<th::Tensor> pos_key_cache
    );

    std::vector<th::Tensor> get_pickle_info() const;

private:
    const at::ScalarType    _st;
    IFZppEncoderModel*      ftzppencoder;
    th::Tensor              head_info;
    th::Tensor              scaling_info;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
