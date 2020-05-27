// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef SRC_COMMONS_COMMONS_HPP_
#define SRC_COMMONS_COMMONS_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <Eigen/Dense>


namespace commons {

template<typename T>
inline const T Norm(const T& val, const T& min_val, const T& max_val) {
  return (val - min_val) / (max_val-min_val);
}

}  // namespace commons

#endif  // SRC_COMMONS_COMMONS_HPP_
