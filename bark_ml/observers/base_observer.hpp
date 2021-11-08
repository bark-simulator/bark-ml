// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_OBSERVERS_BASE_OBSERVER_HPP_
#define BARK_ML_OBSERVERS_BASE_OBSERVER_HPP_

#include <memory>
#include <vector>
#include <tuple>
#include <map>
#include <functional>
#include <Eigen/Dense>

#include "bark/commons/params/params.hpp"
#include "bark/world/world.hpp"
#include "bark/world/observed_world.hpp"
#include "bark/world/goal_definition/goal_definition_state_limits_frenet.hpp"
#include "bark/models/dynamic/dynamic_model.hpp"
#include "bark_ml/commons/spaces.hpp"
#include "bark_ml/commons/commons.hpp"

namespace bark_ml {
namespace observers {

using bark::commons::ParamsPtr;
using spaces::Box;
using bark::world::WorldPtr;
using bark::world::ObservedWorldPtr;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using State = Eigen::Matrix<double, Eigen::Dynamic, 1>;

/**
 * @brief  Base class for the Observer.
 */
class BaseObserver {
 public:
  explicit BaseObserver(const ParamsPtr& params) :
    params_(params) {}

  virtual ObservedState Observe(
    const ObservedWorldPtr& observed_world) const = 0;
  virtual WorldPtr Reset(const WorldPtr& world) = 0;
  virtual Box<double> ObservationSpace() const = 0;

 private:
  ParamsPtr params_;
};

}  // namespace observers
}  // namespace bark_ml

#endif  // BARK_ML_OBSERVERS_BASE_OBSERVER_HPP_
