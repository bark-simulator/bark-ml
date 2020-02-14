// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
#define SRC_OBSERVERS_NEAREST_OBSERVER_HPP_

#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "modules/commons/params/params.hpp"
#include "modules/world/world.hpp"
#include "modules/world/observed_world.hpp"

namespace observers {
using modules::commons::ParamsPtr;
using modules::world::WorldPtr;
using modules::world::ObservedWorldPtr;
using ObservedState = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

class NearestObserver {
 public:
  explicit NearestObserver(const ParamsPtr& params) :
    params_(params) {}

  ObservedState Observe(const ObservedWorldPtr& world) const {
    // TODO(@hart): return observed space
    // 1. build ego agent frenet system
    // 2. find near agents (n)
    // 3. calculate s and d for every agents
    // 4. concat final state
    ObservedState state(1, 10);
    state.setZero();
    return state;
  }

  WorldPtr Reset(const WorldPtr& world,
    const std::vector<int>& agent_ids) {
    return world;
  }

 private:
  ParamsPtr params_;
};

}  // namespace observers

#endif  // SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
