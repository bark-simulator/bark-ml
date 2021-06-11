// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle
// Tobias Kessler and Mansoor Nasir
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_LOADER_HPP_
#define BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_LOADER_HPP_

#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <memory>
// Logging based on availability of glog
#ifdef GOOGLE_STRIP_LOG
#define LOG_INFO VLOG(4)
#define LOG_ERROR LOG(ERROR)
#else
#define LOG 
#define LOG_INFO std::cout
#define LOG_ERROR std::cerr
#endif

namespace bark_ml {
namespace lib_fqf_iqn_qrdqn {

// Load a pytorch trained network (nn.Module) saved from
// python and perform an inference
class ModelLoader {
 public:
  ModelLoader() 
    : module_loaded_(false) {
  }

  bool LoadModel(const std::string& model_filename) {
    if (module_loaded_) {
      return true; 
    }
    LOG_INFO << "Trying to load model from file: " << model_filename << "\n";
    try {
      module_ = torch::jit::load(model_filename);
      module_loaded_ = true;
    }
    catch (const c10::Error& e) {
      LOG_ERROR << "Error loading the model: " << e.msg();
    }

    return module_loaded_;
  }

  std::vector<float> Inference(std::vector<float> state) {
    if (!module_loaded_)
      throw std::runtime_error("Model not loaded!");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor(state).view({1, state.size()}));

    at::Tensor torch_output;
    try {
      // run the inference
      torch_output = module_.forward(inputs).toTensor();

    } catch (const c10::Error& e) {
      std::cerr << e.msg();
      throw std::runtime_error(e.msg());
    }

    //copy the data from torch tensor to std vector
    std::vector<float> output(torch_output.data_ptr<float>(), torch_output.data_ptr<float>() + torch_output.numel());

    return output;
  }

  double TimeMeasuredInference(std::vector<float> state) {
    auto start = std::chrono::high_resolution_clock::now();
    const auto result = Inference(state);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = finish - start;
    double last_inference_time = elapsed.count();
    return last_inference_time;
  }

 private:
  bool module_loaded_;
  torch::jit::script::Module module_;

};

}  // namespace lib_fqf_iqn_qrdqn
}  // namespace bark_ml

#endif //BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_LOADER_HPP_