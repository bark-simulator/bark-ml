// Copyright (c) 2020 fortiss GmbH
//
// Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#ifndef BARK_ML_PYTHON_WRAPPER_PY_NN_TO_VALUE_CONVERTER_HPP_
#define BARK_ML_PYTHON_WRAPPER_PY_NN_TO_VALUE_CONVERTER_HPP_
#include <memory>

#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter.hpp"

namespace py = pybind11;
using bark_ml::lib_fqf_iqn_qrdqn::NNToValueConverter;
using bark_ml::lib_fqf_iqn_qrdqn::ValueType;
using ValueMap = std::unordered_map<ValueType, std::vector<double>>;

class PyNNToValueConverter : public NNToValueConverter {
 public:
  using NNToValueConverter::NNToValueConverter;

  ValueMap ConvertToValueMap(
                const std::vector<float>& nn_output) const {
  PYBIND11_OVERLOAD_PURE(ValueMap, NNToValueConverter, ConvertToValueMap,
                           nn_output); }

};


#endif  // BARK_ML_PYTHON_WRAPPER_PY_NN_TO_VALUE_CONVERTER_HPP_
