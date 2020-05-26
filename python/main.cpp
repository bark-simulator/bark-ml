// Copyright (c) 2020 Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
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

#include "modules/commons/params/params.hpp"
#include "bark_ml/evaluators/base_evaluator.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/observers/nearest_observer.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/commons/spaces.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace py = pybind11;
using modules::commons::ParamsPtr;
using observers::NearestObserver;
using evaluators::GoalReachedEvaluator;
using spaces::Box;
using spaces::Matrix_t;


namespace pybind11 { namespace detail {
    template <typename... Ts>
    struct type_caster<boost::variant<Ts...>> : variant_caster<boost::variant<Ts...>> {};

    template <>
    struct visit_helper<boost::variant> {
        template <typename... Args>
        static auto call(Args &&...args)
            -> decltype(boost::apply_visitor(std::forward<Args>(args)...)) {
            return boost::apply_visitor(std::forward<Args>(args)...);
        }
    };
}}

void python_observers(py::module m) {
  py::class_<NearestObserver,
              std::shared_ptr<NearestObserver>>(m, "NearestObserver")
    .def(py::init<ParamsPtr>())
    .def("Observe", &NearestObserver::Observe)
    .def("Reset", &NearestObserver::Reset)
    .def_property_readonly(
      "observation_space", &NearestObserver::ObservationSpace);
}

void python_evaluators(py::module m) {
  py::class_<GoalReachedEvaluator,
             std::shared_ptr<GoalReachedEvaluator>>(m, "GoalReachedEvaluator")
    .def(py::init<ParamsPtr>())
    .def("Evaluate", &GoalReachedEvaluator::Evaluate)
    .def("Reset", &GoalReachedEvaluator::Reset);
}

void python_spaces(py::module m) {
  py::class_<Box<float>, std::shared_ptr<Box<float>>>(m, "Box")
    .def(py::init<const Matrix_t<float>&,
                  const Matrix_t<float>&,
                  const std::tuple<int>&>())
    .def_property_readonly("low", &Box<float>::low)
    .def_property_readonly("high", &Box<float>::high)
    .def_property_readonly("shape", &Box<float>::shape);
}

PYBIND11_MODULE(bark_ml_library, m) {
  m.doc() = "Additional cpp entities for bark-ml.";
  python_observers(
    m.def_submodule("observers", "c++ observers"));
  python_evaluators(
    m.def_submodule("evaluators", "c++ evaluators"));
  python_spaces(
    m.def_submodule("spaces", "c++ spaces"));
}
