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
#include "bark/commons/params/setter_params.hpp"
#include "bark_ml/evaluators/base_evaluator.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/observers/nearest_observer.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/commons/spaces.hpp"
#include "bark_ml/python_wrapper/pyobserver.hpp"

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace py = pybind11;
using bark::commons::ParamsPtr;
using bark::commons::SetterParams;
using bark_ml::observers::NearestObserver;
using bark_ml::evaluators::GoalReachedEvaluator;
using bark_ml::spaces::Box;
using bark_ml::spaces::Matrix_t;

py::tuple ParamsToPython(const ParamsPtr& params) {
  return py::make_tuple(params->GetCondensedParamList());
}

ParamsPtr PythonToParams(py::tuple t) {
  const auto param_list = t[0].cast<bark::commons::CondensedParamList>();
  return std::make_shared<SetterParams>(true, param_list);
}


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
  py::class_<BaseObserver, PyObserver,
             std::shared_ptr<BaseObserver>>(m, "BaseObserver")
      .def(py::init<const bark::commons::ParamsPtr&>());

  py::class_<NearestObserver, BaseObserver,
              std::shared_ptr<NearestObserver>>(m, "NearestObserver")
    .def(py::init<ParamsPtr>())
    .def("Observe", &NearestObserver::Observe)
    .def("Reset", &NearestObserver::Reset)
    .def_property_readonly(
      "observation_space", &NearestObserver::ObservationSpace)
    .def(py::pickle(
      [](const NearestObserver& o) {
        // We throw away other information such as last trajectories
        return py::make_tuple(ParamsToPython(o.GetParams()));
      },
      [](py::tuple t) {
        if (t.size() != 1)
          throw std::runtime_error("Invalid behavior model state!");
        return new NearestObserver(PythonToParams(t[0].cast<py::tuple>()));
      }));
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

PYBIND11_MODULE(core, m) {
  m.doc() = "Additional cpp entities for bark-ml.";
  python_observers(
    m.def_submodule("observers", "c++ observers"));
  python_evaluators(
    m.def_submodule("evaluators", "c++ evaluators"));
  python_spaces(
    m.def_submodule("spaces", "c++ spaces"));
}
