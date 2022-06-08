#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_loader/model_loader.hpp"
#include <iostream>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using bark_ml::lib_fqf_iqn_qrdqn::ModelLoader;

PYBIND11_MODULE(pytorch_script_wrapper, m) {
    py::class_<ModelLoader>(m, "ModelLoader")
        .def(py::init<>())
        .def("Inference", &ModelLoader::Inference, "Perform the inference for a given state")
        .def("LoadModel", &ModelLoader::LoadModel, "Loads the torch cpp script model")
        .def("TimeMeasuredInference", &ModelLoader::TimeMeasuredInference);
}
