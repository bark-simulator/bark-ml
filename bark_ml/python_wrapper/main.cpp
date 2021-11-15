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

#include "bark/commons/params/params.hpp"
#include "bark_ml/evaluators/base_evaluator.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/observers/base_observer.hpp"
#include "bark_ml/observers/nearest_observer.hpp"
#include "bark_ml/observers/frenet_observer.hpp"
#include "bark_ml/observers/static_observer.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/commons/spaces.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace py = pybind11;
using bark::commons::ParamsPtr;
using bark_ml::observers::NearestObserver;
using bark_ml::observers::FrenetObserver;
using bark_ml::observers::StaticObserver;
using bark_ml::observers::BaseObserver;
using bark_ml::evaluators::GoalReachedEvaluator;
using bark_ml::spaces::Box;
using bark_ml::spaces::Matrix_t;


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
    .def(py::init<const bark::commons::ParamsPtr&>())
    .def("Observe", &NearestObserver::Observe)
    .def("Reset", &NearestObserver::Reset)
    .def_property_readonly(
      "observation_space", &NearestObserver::ObservationSpace);

  py::class_<FrenetObserver,
             std::shared_ptr<FrenetObserver>>(m, "FrenetObserver")
    .def(py::init<const bark::commons::ParamsPtr&>())
    .def("Observe", &FrenetObserver::Observe)
    .def("Reset", &FrenetObserver::Reset)
    .def_property_readonly(
      "observation_space", &FrenetObserver::ObservationSpace);

  py::class_<StaticObserver,
             std::shared_ptr<StaticObserver>>(m, "StaticObserver")
    .def(py::init<const bark::commons::ParamsPtr&>())
    .def("Observe", &StaticObserver::Observe)
    .def("Reset", &StaticObserver::Reset)
    .def_property_readonly(
      "observation_space", &StaticObserver::ObservationSpace);
}


void python_evaluators(py::module m) {
  py::class_<GoalReachedEvaluator,
             std::shared_ptr<GoalReachedEvaluator>>(m, "GoalReachedEvaluator")
    .def(py::init<const bark::commons::ParamsPtr&>())
    .def("Evaluate", &GoalReachedEvaluator::Evaluate)
    .def("Reset", &GoalReachedEvaluator::Reset);
}

void python_spaces(py::module m) {
  py::class_<Box<double>, std::shared_ptr<Box<double>>>(m, "Box")
    .def(py::init<const Matrix_t<double>&,
                  const Matrix_t<double>&,
                  const std::tuple<int>&>())
    .def_property_readonly("low", &Box<double>::low)
    .def_property_readonly("high", &Box<double>::high)
    .def_property_readonly("shape", &Box<double>::shape);
}

PYBIND11_MODULE(core, m) {
  m.doc() = "Additional cpp entities for bark-ml.";
  python_observers(
    m.def_submodule("observers", "c++ observers"));
  python_evaluators(
    m.def_submodule("evaluators", "c++ evaluators"));
  python_spaces(
    m.def_submodule("spaces", "c++ spaces"));
}
