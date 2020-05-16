// Copyright (c) Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_OBSERVERS_NEAREST_OBSERVER_HPP_
#define BARK_ML_OBSERVERS_NEAREST_OBSERVER_HPP_

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
using modules::world::Agent;
using spaces::Box;
using commons::Norm;
using spaces::Matrix_t;
using modules::world::AgentMap;
using modules::world::AgentPtr;
using modules::world::WorldPtr;
using modules::world::goal_definition::GoalDefinitionStateLimitsFrenet;
using modules::world::ObservedWorldPtr;
using modules::geometry::Point2d;
using modules::geometry::Line;
using modules::geometry::Distance;
using modules::models::dynamic::StateDefinition;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using modules::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;


class NearestObserver {
 public:
  explicit NearestObserver(const ParamsPtr& params) :
    params_(params) {
      nearest_agent_num_ =
        params_->GetInt(
          "ML::Observer::n_nearest_agents", "Nearest agents number", 4);
      min_theta_ = params_->GetReal("ML::Observer::min_theta", "", -3.14);
      max_theta_ = params_->GetReal("ML::Observer::max_theta", "", 3.14);
      min_vel_ = params_->GetReal("ML::Observer::min_vel", "", 0.0);
      max_vel_ = params_->GetReal("ML::Observer::max_vel", "", 25.0);
      max_dist_ = params_->GetReal("ML::Observer::max_dist", "", 75.0);
      state_size_ = params_->GetInt("ML::Observer::state_size", "", 4);
      observation_len_ = nearest_agent_num_ * state_size_;
  }

  // TODO(@hart): add NormState fct. or integrate in this fct.
  ObservedState FilterState(const State& state) const {
    ObservedState ret_state(1, state_size_);
    ret_state << state(StateDefinition::X_POSITION),
                 state(StateDefinition::Y_POSITION),
                 state(StateDefinition::THETA_POSITION),
                 state(StateDefinition::VEL_POSITION);
    return ret_state;
  }

  ObservedState Observe(const ObservedWorldPtr& observed_world) const {
    int row_idx = 0;
    ObservedState state(1, observation_len_);
    state.setZero();

    // find near agents (n)
    AgentMap nearest_agents = observed_world->GetNearestAgents(
      observed_world->CurrentEgoPosition(), nearest_agent_num_);

    // sort agents by distance and distance < max_dist_
    std::map<float, AgentPtr, std::greater<float>> distance_agent_map;
    for (const auto& agent : nearest_agents) {
      const auto& agent_state = agent.second->GetCurrentPosition();
      float distance = Distance(
        observed_world->CurrentEgoPosition(), agent_state);
      if (distance < max_dist_)
        distance_agent_map[distance] = agent.second;
    }

    // add ego agent state
    ObservedState obs_ego_agent_state =
      FilterState(observed_world->CurrentEgoState());
    state.block(0, row_idx*state_size_, 1, state_size_) = obs_ego_agent_state;
    row_idx++;

    // add other states
    for (const auto& agent : distance_agent_map) {
      if (agent.second->GetAgentId() != observed_world->GetEgoAgentId()) {
        ObservedState other_agent_state =
          FilterState(agent.second->GetCurrentState());
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
  int state_size_, nearest_agent_num_, observation_len_;
  float min_theta_, max_theta_, min_vel_, max_vel_, max_dist_;
};

}  // namespace observers

#endif  // BARK_ML_OBSERVERS_NEAREST_OBSERVER_HPP_
