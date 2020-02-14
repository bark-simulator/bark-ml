// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
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

#include "src/observers/nearest_observer.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace py = pybind11;

PYBIND11_MODULE(bark_ml, m) {
  m.doc() = "Wrapper for bark-ml.";
  using observers::NearestObserver;

  void python_observers(py::module m) {
    py::class_<NearestObserver,
               std::shared_ptr<NearestObserver>>(m, "NearestObserver")
      .def(py::init<ParamsPtr>())
      .def("observe", &NearestObserver::Observe);
      .def("reset", &NearestObserver::Reset);
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
