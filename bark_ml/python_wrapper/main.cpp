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
#include "bark_ml/observers/base_observer.hpp"
#include "bark_ml/observers/nearest_observer.hpp"
#include "bark_ml/observers/frenet_observer.hpp"
#include "bark_ml/observers/static_observer.hpp"
#include "bark_ml/evaluators/goal_reached.hpp"
#include "bark_ml/commons/spaces.hpp"
#include "bark_ml/python_wrapper/pyobserver.hpp"
#include "bark_ml/python_wrapper/pynn_to_value_converter.hpp"
#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter_sequential.hpp"
#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter_policy.hpp"


PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

namespace py = pybind11;
using bark::commons::ParamsPtr;
using bark::commons::SetterParams;
using bark_ml::observers::NearestObserver;
using bark_ml::observers::FrenetObserver;
using bark_ml::observers::StaticObserver;
using bark_ml::observers::BaseObserver;
using bark_ml::evaluators::GoalReachedEvaluator;
using bark_ml::spaces::Box;
using bark_ml::spaces::Matrix_t;
using namespace bark_ml::lib_fqf_iqn_qrdqn;

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
        return py::make_tuple(ParamsToPython(o.GetParams()));
      },
      [](py::tuple t) {
        if (t.size() != 1)
          throw std::runtime_error("Invalid behavior model state!");
        return new NearestObserver(PythonToParams(t[0].cast<py::tuple>()));
      }));

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

void python_value_converters(py::module m) {
  py::class_<NNToValueConverter, PyNNToValueConverter,
             std::shared_ptr<NNToValueConverter>>(m, "NNToValueConverter")
      .def(py::init<const unsigned&>());

  py::class_<NNToValueConverterSequential, NNToValueConverter,
            std::shared_ptr<NNToValueConverterSequential>>(m, "NNToValueConverterSequential")
  .def(py::init<const unsigned&>())
  .def(py::pickle(
    [](const NNToValueConverterSequential& nn) {
      return py::make_tuple(nn.GetNumActions());
    },
    [](py::tuple t) {
      if (t.size() != 1)
        throw std::runtime_error("Invalid NNToValueConverterSequential state!");
      return new NNToValueConverterSequential(t[0].cast<unsigned>());
    }));

  py::class_<NNToValueConverterPolicy, NNToValueConverter,
            std::shared_ptr<NNToValueConverterPolicy>>(m, "NNToValueConverterPolicy")
  .def(py::init<const unsigned&>())
  .def(py::pickle(
    [](const NNToValueConverterPolicy& nn) {
      return py::make_tuple(nn.GetNumActions());
    },
    [](py::tuple t) {
      if (t.size() != 1)
        throw std::runtime_error("Invalid NNToValueConverterPolicy state!");
      return new NNToValueConverterPolicy(t[0].cast<unsigned>());
    }));
}

PYBIND11_MODULE(core, m) {
  m.doc() = "Additional cpp entities for bark-ml.";
  python_observers(
    m.def_submodule("observers", "c++ observers"));
  python_evaluators(
    m.def_submodule("evaluators", "c++ evaluators"));
  python_spaces(
    m.def_submodule("spaces", "c++ spaces"));
  python_value_converters(
    m.def_submodule("value_converters", "value conversion"));
}
