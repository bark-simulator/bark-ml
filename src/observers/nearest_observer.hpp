// Copyright (c) 2019 fortiss GmbH, Julian Bernhard, Klemens Esterle, Patrick Hart, Tobias Kessler
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

namespace observers {
using modules::commons::ParamsPtr;
using modules::world::WorldPtr;
using ObservedState = Eigen::Matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>;

class NearestObserver {
 public:
  explicit NearestObserver(const ParamsPtr& params) :
    params_(params) {}

  ObservedState Observe(const WorldPtr& world,
    const std::vector<int>& agent_ids) const {
    // TODO(@hart): return observed space
  }

  WorldPtr Reset(const WorldPtr& world,
    const std::vector<int>& agent_ids) {
    return world;
  }

 private:
  ParamsPtr params_;
}

}  // namespace observers

#endif  // SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
