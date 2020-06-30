// Copyright (c) 2020 Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_COMMONS_COMMONS_HPP_
#define BARK_ML_COMMONS_COMMONS_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <Eigen/Dense>


namespace bark_ml {
namespace commons {

template<typename T>
inline const T Norm(const T& val, const T& min_val, const T& max_val) {
  return (val - min_val) / (max_val-min_val);
}

}  // namespace commons
}  // namespace bark_ml

#endif  // BARK_ML_COMMONS_COMMONS_HPP_
