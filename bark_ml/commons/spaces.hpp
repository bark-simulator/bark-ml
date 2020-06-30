// Copyright (c) 2020 Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_COMMONS_SPACES_HPP_
#define BARK_ML_COMMONS_SPACES_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <Eigen/Dense>


namespace bark_ml {
namespace spaces {

template<typename T>
using Matrix_t = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template<typename T>
struct Box {
  Box(const Matrix_t<T>& low,
      const Matrix_t<T>& high,
      const std::tuple<int>& shape) :
    low_(low), high_(high), shape_(shape) {}
  Matrix_t<T> low() const {
    return low_;
  }
  Matrix_t<T> high() const {
    return high_;
  }
  std::tuple<int> shape() {
    return shape_;
  }
  const Matrix_t<T> low_;
  const Matrix_t<T> high_;
  const std::tuple<int> shape_;
};

}  // namespace spaces
}  // namespace bark_ml

#endif  // BARK_ML_COMMONS_SPACES_HPP_
