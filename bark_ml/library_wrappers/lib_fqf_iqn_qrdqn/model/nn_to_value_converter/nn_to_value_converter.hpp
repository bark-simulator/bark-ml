


// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle
// Tobias Kessler and Mansoor Nasir
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_HPP_
#define BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_HPP_

#include <unordered_map>
#include <vector>
#include <memory>

#ifdef GOOGLE_STRIP_LOG
#define LOG_INFO VLOG(4)
#define LOG_ERROR LOG(ERROR)
#define LOG_FATAL LOG(FATAL)
#else
#define LOG 
#define LOG_INFO std::cout
#define LOG_ERROR std::cerr
#define LOG_FATAL std::cerr
#endif


namespace bark_ml {
namespace lib_fqf_iqn_qrdqn {

typedef enum ValueType {
    Return = 0,
    EnvelopeRisk = 1,
    CollisionRisk = 2
} ValueType;

class NNToValueConverter {
 public:
  NNToValueConverter(const unsigned& num_actions) : num_actions_(num_actions) {}

  virtual ~NNToValueConverter() {}

  virtual std::unordered_map<ValueType, std::vector<double>> ConvertToValueMap(const std::vector<float>& nn_output) const = 0;

  unsigned GetNumActions() const { return num_actions_; }
private:
 unsigned num_actions_;
};

typedef std::shared_ptr<NNToValueConverter> NNToValueConverterPtr;

}  // namespace lib_fqf_iqn_qrdqn
}  // namespace bark_ml

#endif //BARK_ML_LIB_FQF_IQN_QRDQN_MODEL_NN_TO_VALUE_CONVERTER_NN_TO_VALUE_CONVERTER_HPP_