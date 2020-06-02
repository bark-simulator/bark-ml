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
#include "bark_ml/observers/base_observer.hpp"

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
using modules::geometry::Norm0To2PI;
using modules::models::dynamic::StateDefinition::X_POSITION;
using modules::models::dynamic::StateDefinition::Y_POSITION;
using modules::models::dynamic::StateDefinition::THETA_POSITION;
using modules::models::dynamic::StateDefinition::VEL_POSITION;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using modules::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;


class NearestObserver : public BaseObserver {
 public:
  explicit NearestObserver(const ParamsPtr& params) :
    BaseObserver(params),
    min_x_(0.), max_x_(100.),
    min_y_(0.), max_y_(100.),
    min_theta_(0.), max_theta_(2*3.14) {
      nearest_agent_num_ =
        params->GetInt(
          "ML::NearestObserver::NNearestAgents", "Nearest agents number", 4);
      min_vel_ = params->GetReal("ML::NearestObserver::MinVel", "", 0.0);
      max_vel_ = params->GetReal("ML::NearestObserver::MaxVel", "", 50.0);
      max_dist_ = params->GetReal("ML::NearestObserver::MaxDist", "", 75.0);
      state_size_ = params->GetInt("ML::NearestObserver::StateSize", "", 4);
      observation_len_ = nearest_agent_num_ * state_size_;
  }

  float Norm(const float val, const float mi, const float ma) const {
    return (val - mi)/(ma - mi);
  }

  ObservedState FilterState(const State& state) const {
    ObservedState ret_state(1, state_size_);
    const float normalized_angle = Norm0To2PI(state(THETA_POSITION));
    ret_state << Norm(state(X_POSITION), min_x_, max_x_),
                 Norm(state(Y_POSITION), min_y_, max_y_),
                 Norm(normalized_angle, min_theta_, max_theta_),
                 Norm(state(VEL_POSITION), min_vel_, max_vel_);
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
        state.block(0, row_idx*state_size_, 1, state_size_) = other_agent_state;  // NOLINT
        row_idx++;
      }
    }
    return state;
  }

  WorldPtr Reset(const WorldPtr& world) {
    const auto& x_y = world->BoundingBox();
    Point2d bb0 = x_y.first;
    Point2d bb1 = x_y.second;
    min_x_ = bb0.get<0>();
    max_x_ = bb1.get<0>();
    min_y_ = bb0.get<1>();
    max_y_ = bb1.get<1>();
    return world;
  }

  Box<float> ObservationSpace() const {
    Matrix_t<float> low(1, observation_len_);
    low.setZero();
    Matrix_t<float> high(1, observation_len_);
    high.setOnes();
    std::tuple<int> shape{observation_len_};
    return Box<float>(low, high, shape);
  }

 private:
  int state_size_, nearest_agent_num_, observation_len_;
  float min_theta_, max_theta_, min_vel_, max_vel_, max_dist_,
         min_x_, max_x_, min_y_, max_y_;
};

}  // namespace observers

#endif  // BARK_ML_OBSERVERS_NEAREST_OBSERVER_HPP_
