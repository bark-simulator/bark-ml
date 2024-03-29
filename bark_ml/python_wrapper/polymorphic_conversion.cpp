// Copyright (c) 2020 Julian Bernhard
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "bark_ml/python_wrapper/polymorphic_conversion.hpp"
#include "bark_ml/observers/nearest_observer.hpp"
#include "bark/commons/params/setter_params.hpp"
#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter_sequential.hpp"
#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter_policy.hpp"

namespace py = pybind11;

using namespace bark_ml::observers;
using namespace bark_ml::lib_fqf_iqn_qrdqn;

py::tuple ObserverToPython(ObserverPtr observer) {
  std::string observer_name;
  if (typeid(*observer) == typeid(NearestObserver)) {
    observer_name = "NearestObserver";
  } else {
    LOG(FATAL) << "Unknown Observer for polymorphic conversion to python: "
               << typeid(*observer).name();
  }
  return py::make_tuple(observer, observer_name);
}

ObserverPtr PythonToObserver(py::tuple t) {
  std::string observer_name = t[1].cast<std::string>();
  if (observer_name.compare("NearestObserver") == 0) {
    return std::make_shared<NearestObserver>(
        t[0].cast<NearestObserver>());
  }  else {
       LOG(FATAL) << "Unknown Observer for polymorphic conversion to cpp: "
               << observer_name;
  }
}


py::tuple NNToValueConverterToPython(NNToValueConverterPtr converter) {
  std::string converter_name;
  if (typeid(*converter) == typeid(NNToValueConverterSequential)) {
    converter_name = "NNToValueConverterSequential";
  } else if (typeid(*converter) == typeid(NNToValueConverterPolicy)) {
    converter_name = "NNToValueConverterPolicy";
  } else {
    LOG(FATAL) << "Unknown Converter for polymorphic conversion to python: "
               << typeid(*converter).name();
  }
  return py::make_tuple(converter, converter_name);
}
NNToValueConverterPtr PythonToNNToValueConverter(py::tuple t) {
  std::string converter_name = t[1].cast<std::string>();
  if (converter_name.compare("NNToValueConverterSequential") == 0) {
    return std::make_shared<NNToValueConverterSequential>(
        t[0].cast<NNToValueConverterSequential>());
  } else if (converter_name.compare("NNToValueConverterPolicy") == 0) {
    return std::make_shared<NNToValueConverterPolicy>(
        t[0].cast<NNToValueConverterPolicy>());
  } else {
       LOG(FATAL) << "Unknown Converter for polymorphic conversion to cpp: "
               << converter_name;
  }
}