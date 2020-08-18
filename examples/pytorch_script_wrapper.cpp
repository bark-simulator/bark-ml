#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Load a pytorch trained network (nn.Module) saved from
// python and perform an inference
class ModelLoader {
public:
  ModelLoader(const std::string& model_filename, const long action_space, const long state_space) 
    : model_filename_(model_filename), module_loaded_(false),action_space_(action_space), state_space_(state_space) {
  }

bool LoadModel() {
  if (module_loaded_) {
    return true; // module already loaded
  }

  // load the torch script module saved from python
  try {
    module_ = torch::jit::load(model_filename_);
    module_loaded_ = true;
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
  }

  return module_loaded_;
}

std::vector<float> Inference(std::vector<float> state) {
  std::vector<float> actions(action_space_);

  if(module_loaded_) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor(state).view({1, state_space_}));

    at::Tensor predicted_actions;
    try {
      // run the inference
      predicted_actions = module_.forward(inputs).toTensor();

    } catch (const c10::Error& e) {
      std::cerr << e.msg();
      return actions;
    }

    //copy the data from torch tensor to std vector
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tharray = torch::from_blob(actions.data(), {1, action_space_}, options);
    tharray.copy_(predicted_actions);
  }

  return actions;
}

private:
std::string model_filename_;
long action_space_;
long state_space_;
bool module_loaded_;
torch::jit::script::Module module_;

};


// PYBIND11_MODULE(pytorch_script_wrapper, m) {
//     m.doc() = "pybind11 example plugin"; // optional module docstring

//     m.def("predict", &ModelLoader::Inference, "A function which predicts action in a given state");
// }


PYBIND11_MODULE(pytorch_script_wrapper, m) {
    py::class_<ModelLoader>(m, "ModelLoader")
        .def(py::init<const std::string &, const long, const long>())
        .def("Inference", &ModelLoader::Inference, "Perform the inference for a given state")
        .def("LoadModel", &ModelLoader::LoadModel, "Loads the torch cpp script model");
}
