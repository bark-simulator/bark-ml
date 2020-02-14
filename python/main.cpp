// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include <chrono>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/stl_bind.h"
#include "pybind11/eigen.h"
#include "boost/variant.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace py = pybind11;

PYBIND11_MODULE(bark_ml, m) {
  m.doc() = "Wrapper for bark-ml.";

  void python_observers(py::module m) {
    // py::class_<World, std::shared_ptr<World>>(m, "NearestObserver")
    //   .def(py::init<ParamsPtr>());
  }

  void python_evaluators(py::module m) {
    // py::class_<World, std::shared_ptr<World>>(m, "NearestObserver")
    //   .def(py::init<ParamsPtr>());
  }

  python_observers(
    m.def_submodule("observers", "c++ observers"));
  python_evaluators(
    m.def_submodule("evaluators", "c++ observers"));
}
