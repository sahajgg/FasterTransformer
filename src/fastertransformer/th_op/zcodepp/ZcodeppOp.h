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
    virtual void forward(size_t batch_size,
                        size_t seq_len,
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

        static std::string cublas_config_path = "gemm_config.in";
        static char* generated_config_path = std::getenv("FT_ZCODEPP_GEMM_CONFIG_PATH");
        if (generated_config_path != nullptr) {
            cublas_config_path = std::string(generated_config_path);
        }

        cublas_algo_map_            = new ft::cublasAlgoMap(cublas_config_path, "");
        cublas_wrapper_mutex_       = new std::mutex();

        // Model-level weights
        zppencoder_weights.word_embedding_table                       = get_ptr<T>(_weights[0]);
        zppencoder_weights.word_embedding_layernorm_weights.gamma     = get_ptr<T>(_weights[1]);
        zppencoder_weights.word_embedding_layernorm_weights.beta      = get_ptr<T>(_weights[2]);

        // Layer-level weights
        zppencoder_weights.zpp_encoder_layer_weights.clear();
        zppencoder_weights.zpp_encoder_layer_weights.resize(_layer_num);
        for (int i = 0; i < _layer_num; i++) {
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.query_weight.kernel = get_ptr<T>(_weights[3 + (_layer_num * i) + 0]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.query_weight.bias = get_ptr<T>(_weights[3 + (_layer_num * i) + 1]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.key_weight.kernel = get_ptr<T>(_weights[3 + (_layer_num * i) + 2]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.key_weight.bias = get_ptr<T>(_weights[3 + (_layer_num * i) + 3]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.value_weight.kernel = get_ptr<T>(_weights[3 + (_layer_num * i) + 4]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.value_weight.bias = get_ptr<T>(_weights[3 + (_layer_num * i) + 5]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.attention_output_weight.kernel = get_ptr<T>(_weights[3 + (_layer_num * i) + 6]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attention_weights.attention_output_weight.bias = get_ptr<T>(_weights[3 + (_layer_num * i) + 7]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attn_layernorm_weights.gamma = get_ptr<T>(_weights[3 + (_layer_num * i) + 8]);
            zppencoder_weights.zpp_encoder_layer_weights[i].attn_layernorm_weights.beta = get_ptr<T>(_weights[3 + (_layer_num * i) + 9]);
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.intermediate_weight.kernel = get_ptr<T>(_weights[3 + (_layer_num * i) + 10]);
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.intermediate_weight.bias = get_ptr<T>(_weights[3 + (_layer_num * i) + 11]);
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.output_weight.kernel = get_ptr<T>(_weights[3 + (_layer_num * i) + 12]);
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_weights.output_weight.bias = get_ptr<T>(_weights[3 + (_layer_num * i) + 13]);
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_layernorm_weights.gamma = get_ptr<T>(_weights[3 + (_layer_num * i) + 14]);
            zppencoder_weights.zpp_encoder_layer_weights[i].ffn_layernorm_weights.beta = get_ptr<T>(_weights[3 + (_layer_num * i) + 15]);
        }
    }

    ~FTZppEncoderModel() override
    {
        cublasLtDestroy(_cublasltHandle);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(size_t batch_size,
                 size_t seq_len,
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

        ft::TensorMap input_tensors_map = ft::TensorMap(
            {
                {"input_ids", ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size, seq_len}, get_ptr<int>(input)}}, 
                {"sequence_lengths", ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{batch_size}, get_ptr<int>(sequence_lengths)}}
            }
        );

        ft::TensorMap pos_query_cache_map, pos_key_cache_map;
        for (uint l = 0; l < _layer_num; l++) {
            pos_query_cache_map.insert(
                "pos_query_cache_" + std::to_string(l), 
                ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, _head_num, 2 * _position_buckets, _head_size}, get_ptr<T>(pos_query_cache[l])}
            );
            pos_key_cache_map.insert(
                "pos_key_cache_" + std::to_string(l), 
                ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, _head_num, 2 * _position_buckets, _head_size}, get_ptr<T>(pos_key_cache[l])}
            );
        }

        ft::TensorMap output_tensors_map = ft::TensorMap(
            {
                {"output_hidden_state", ft::Tensor{ft::MEMORY_GPU, data_type, std::vector<size_t>{batch_size, seq_len, (size_t)(_head_num * _head_size)}, get_ptr<T>(output)}}
            }
        );

        try {
            zppencoder->forward(
                &output_tensors_map, 
                &input_tensors_map,
                &pos_query_cache_map,
                &pos_key_cache_map,
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
    FasterTransformerZppEncoderModel(std::vector<th::Tensor> model_weights,
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
