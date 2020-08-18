#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Load a pytorch trained network (nn.Module) saved from
// python and perform an inference


std::vector<float> predict(std::string script_model_path, std::vector<float> state) {

  std::vector<float> actions(50);

  torch::jit::script::Module module;
  try {
    // deserialize the ScriptModule from a saved python trained model using torch::jit::load().
    module = torch::jit::load(script_model_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return actions;
  }
  

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::tensor(state).view({1,16}));
  
  at::Tensor actions2;
  try {
    // run the inference
    actions2 = module.forward(inputs).toTensor();

  } catch (const c10::Error& e) {
    std::cerr << e.msg();
    return actions;
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor tharray = torch::from_blob(actions.data(), {1,50}, options);
  //tharray.copy_(actions2);
  tharray.copy_(actions2);
  //export LD_LIBRARY_PATH=/home/mansoor/Study/Werkstudent/fortiss/code/bark-ml/bark_ml/python_wrapper/venv/lib/python3.7/site-packages/torch/lib/

  return actions;
}


PYBIND11_MODULE(pytorch_script_wrapper, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("predict", &predict, "A function which predicts action in a given state");
}
