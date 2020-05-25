// Copyright (c) Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
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

#include "modules/commons/params/params.hpp"
#include "modules/world/world.hpp"
#include "modules/world/observed_world.hpp"
#include "modules/world/goal_definition/goal_definition_state_limits_frenet.hpp"
#include "modules/models/dynamic/dynamic_model.hpp"
#include "bark_ml/commons/spaces.hpp"
#include "bark_ml/commons/commons.hpp"

namespace observers {
using modules::commons::ParamsPtr;
using spaces::Box;
using modules::world::WorldPtr;
using modules::world::ObservedWorldPtr;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;

class BaseObserver {
 public:
  explicit BaseObserver(const ParamsPtr& params) :
    params_(params) {}

  virtual ObservedState Observe(
    const ObservedWorldPtr& observed_world) const = 0;
  virtual WorldPtr Reset(const WorldPtr& world) = 0;
  virtual Box<float> ObservationSpace() const = 0;

 private:
  ParamsPtr params_;
};

}  // namespace observers

#endif  // BARK_ML_OBSERVERS_BASE_OBSERVER_HPP_
