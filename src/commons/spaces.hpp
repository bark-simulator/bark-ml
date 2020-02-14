// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef SRC_COMMONS_SPACES_HPP_
#define SRC_COMMONS_SPACES_HPP_

#include <memory>
#include <vector>
#include <Eigen/Dense>


namespace spaces {

template<typename T>
using Matrix_t = Eigen::Matrix_t<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
struct Box {
  Box(const Matrix_t<T>& low,
      const Matrix_t<T>& high,
      const Matrix_t<T>& shape) :
    low_(low), high_(high), shape_(shape) {}
  const Matrix_t<T> low_;
  const Matrix_t<T> high_;
  const Matrix_t<T> shape_;
};

}  // namespace spaces

#endif  // SRC_COMMONS_SPACES_HPP_
