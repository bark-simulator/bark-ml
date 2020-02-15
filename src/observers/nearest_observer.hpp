// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
#define SRC_OBSERVERS_NEAREST_OBSERVER_HPP_

#include <memory>
#include <vector>
#include <tuple>
#include <Eigen/Dense>

#include "modules/commons/params/params.hpp"
#include "modules/world/world.hpp"
#include "modules/world/observed_world.hpp"
#include "modules/models/dynamic/dynamic_model.hpp"
#include "src/commons/spaces.hpp"

namespace observers {
using modules::commons::ParamsPtr;
using modules::world::Agent;
using spaces::Box;
using spaces::Matrix_t;
using modules::world::AgentMap;
using modules::world::WorldPtr;
using modules::world::ObservedWorldPtr;
using modules::geometry::Point2d;
using modules::geometry::Line;
using modules::models::dynamic::StateDefinition;
using ObservedState = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using modules::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;

class NearestObserver {
 public:
  explicit NearestObserver(const ParamsPtr& params) :
    params_(params),
    state_size_(4) {
      nearest_agent_num_ =
        params_->GetInt("ML::Observer::n_nearest_agents",
                        "Nearest agents number", 4);
      observation_len_ = nearest_agent_num_ * state_size_;
  }

  ObservedState TransformState(
    const State& state, const Line& center_line) const {
    Point2d pose(
      state(StateDefinition::X_POSITION),
      state(StateDefinition::Y_POSITION));
    FrenetPosition frenet_pos(pose, center_line);
    ObservedState ret_state(1, state_size_);
    ret_state << frenet_pos.lon,
                 frenet_pos.lat,
                 state(StateDefinition::THETA_POSITION),
                 state(StateDefinition::VEL_POSITION);
    return ret_state;
  }

  ObservedState Observe(const ObservedWorldPtr& world) const {
    int row_idx = 0;
    ObservedState state(1, observation_len_);
    state.setZero();
    // TODO(@hart): this should later be removed
    world->UpdateAgentRTree();

    // 1. build ego agent frenet system
    std::shared_ptr<const Agent> ego_agent = world->GetEgoAgent();
    const Point2d ego_pos = ego_agent->GetCurrentPosition();
    const auto& ego_lane_corridor =
      ego_agent->GetRoadCorridor()->GetCurrentLaneCorridor(ego_pos);

    // 2. find near agents (n)
    AgentMap nearest_agents = world->GetNearestAgents(ego_pos, 5);

    // ego agent state
    ObservedState obs_ego_agent_state =
      TransformState(ego_agent->GetCurrentState(),
                     ego_lane_corridor->GetCenterLine());
    state.block(0, row_idx*state_size_, 1, state_size_) = obs_ego_agent_state;
    row_idx++;

    // other states
    for (const auto& agent : nearest_agents) {
      if (agent.second->GetAgentId() != ego_agent->GetAgentId()) {
        ObservedState other_agent_state =
          TransformState(agent.second->GetCurrentState(),
                         ego_lane_corridor->GetCenterLine());
        state.block(0, row_idx*state_size_, 1, state_size_) = other_agent_state;
        row_idx++;
      }
    }

    return state;
  }

  WorldPtr Reset(const WorldPtr& world,
    const std::vector<int>& agent_ids) {
    return world;
  }

  Box<float> ObservationSpace() const {
    Matrix_t<float> low(1, observation_len_);
    low.setZero();
    Matrix_t<float> high(1, observation_len_);
    high.setOnes();
    std::vector<int> shape{1, observation_len_};
    return Box<float>(low, high, shape);
  }

 private:
  ParamsPtr params_;
  const int state_size_;
  int nearest_agent_num_;
  int observation_len_;
};

}  // namespace observers

#endif  // SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
