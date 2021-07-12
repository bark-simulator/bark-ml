


// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle
// Tobias Kessler and Mansoor Nasir
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_POLICY_HPP_
#define BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_POLICY_HPP_

#include <math.h>
#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter.hpp"


namespace bark_ml {
namespace lib_fqf_iqn_qrdqn {

class NNToValueConverterPolicy : public NNToValueConverter {
 public:
  NNToValueConverterPolicy(const unsigned& num_actions) : NNToValueConverter(num_actions) {}

  virtual ~NNToValueConverterPolicy() {}

  virtual std::unordered_map<ValueType, std::vector<double>> ConvertToValueMap(const std::vector<float>&  nn_output) const {
    std::unordered_map<ValueType, std::vector<double>> value_map;

    if(nn_output.size() != num_actions_) {
      LOG_FATAL << "Invalid neural network output with size of " << nn_output.size() << " for num_actions = " << num_actions_;
    }
    // Only a policy output for this conversion type
    value_map[ValueType::Policy] = std::vector<double>(nn_output.begin(), nn_output.begin()+num_actions_);
    
    return value_map;
  };

};

}  // namespace lib_fqf_iqn_qrdqn
}  // namespace bark_ml

#endif //BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_POLICY_HPP_