// Copyright (c) 2020 Julian Bernhard
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#ifndef BARK_OBSERVERS_POLYMORPHIC_CONVERSION_HPP_
#define BARK_OBSERVERS_POLYMORPHIC_CONVERSION_HPP_

#include "bark/python_wrapper/common.hpp"
#include "bark_ml/observers/base_observer.hpp"
#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter.hpp"

namespace py = pybind11;

// For pickle we need conversion functions between the genereric base types and
// the derived types

py::tuple ObserverToPython(bark_ml::observers::ObserverPtr observer);
bark_ml::observers::ObserverPtr PythonToObserver(py::tuple t);

py::tuple NNToValueConverterToPython(bark_ml::lib_fqf_iqn_qrdqn::NNToValueConverterPtr converter);
bark_ml::lib_fqf_iqn_qrdqn::NNToValueConverterPtr PythonToNNToValueConverter(py::tuple t);

#endif  // BARK_OBSERVERS_POLYMORPHIC_CONVERSION_HPP_
