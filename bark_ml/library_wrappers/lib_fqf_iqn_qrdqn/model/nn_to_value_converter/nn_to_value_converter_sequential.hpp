


// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle
// Tobias Kessler and Mansoor Nasir
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_SEQUENTIAL_HPP_
#define BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_SEQUENTIAL_HPP_

#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter/nn_to_value_converter.hpp"


namespace bark_ml {
namespace lib_fqf_iqn_qrdqn {

class NNToValueConverterSequential : public NNToValueConverter {
 public:
  NNToValueConverterSequential(const unsigned& num_actions) : NNToValueConverter(num_actions) {}

  virtual ~NNToValueConverterSequential() {}

  virtual std::unordered_map<ValueType, std::vector<double>> ConvertToValueMap(const std::vector<float>&  nn_output) const {
    std::unordered_map<ValueType, std::vector<double>> value_map;

    if(nn_output.size() != num_actions_*3) {
      LOG_FATAL << "Invalid neural network output with size of " << nn_output.size() << " for num_actions = " << num_actions_;
    }
    // Convention that first num_actions elements are return 
    value_map[ValueType::EnvelopeRisk] = std::vector<double>(nn_output.begin(), nn_output.begin()+num_actions_);
    //... second num action elements are envelope risk
    value_map[ValueType::CollisionRisk] = std::vector<double>(nn_output.begin()+num_actions_, nn_output.begin()+2*num_actions_);
    // ... and third num action elements are collision risk
    value_map[ValueType::Return] = std::vector<double>(nn_output.begin()+2*num_actions_, nn_output.begin()+3*num_actions_);

    return value_map;
  };

};

}  // namespace lib_fqf_iqn_qrdqn
}  // namespace bark_ml

#endif //BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_SEQUENTIAL_HPP_