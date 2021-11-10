// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_OBSERVERS_FRENET_OBSERVER_HPP_
#define BARK_ML_OBSERVERS_FRENET_OBSERVER_HPP_

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
#include "bark_ml/observers/base_observer.hpp"

namespace bark_ml {
namespace observers {

using bark::commons::ParamsPtr;
using bark::world::Agent;
using spaces::Box;
using commons::Norm;
using spaces::Matrix_t;
using bark::world::AgentMap;
using bark::world::AgentPtr;
using bark::world::objects::AgentId;
using bark::world::WorldPtr;
using bark::world::goal_definition::GoalDefinitionStateLimitsFrenet;
using bark::world::map::LaneCorridorPtr;
using bark::world::ObservedWorldPtr;
using bark::world::ObservedWorld;
using bark::geometry::Point2d;
using bark::commons::transformation::FrenetState;
using bark::commons::transformation::FrenetStateDifference;
using bark::geometry::B_2PI;
using bark::geometry::B_PI;
using bark::geometry::Line;
using bark::geometry::Distance;
using bark::geometry::NormToPI;
using bark::models::dynamic::StateDefinition::X_POSITION;
using bark::models::dynamic::StateDefinition::Y_POSITION;
using bark::models::dynamic::StateDefinition::THETA_POSITION;
using bark::models::dynamic::StateDefinition::VEL_POSITION;
using bark::commons::transformation::FrenetPosition;


class FrenetObserver {
 public:
  explicit FrenetObserver(const ParamsPtr& params) {
      nearest_agent_num_ =
        params->GetInt(
          "ML::FrenetObserver::NNearestAgents", "Nearest agents number", 2);
      min_vel_lon_ = params->GetReal("ML::FrenetObserver::MinVelLon", "", -20.0);
      max_vel_lon_ = params->GetReal("ML::FrenetObserver::MaxVelLon", "", 20.0);
      min_vel_lat_ = params->GetReal("ML::FrenetObserver::MinVelLat", "", -10.0);
      max_vel_lat_ = params->GetReal("ML::FrenetObserver::MaxVelLat", "", 10.0);
      max_dist_ = params->GetReal("ML::FrenetObserver::MaxDist", "", 75.0);
      min_s_ = params->GetReal("ML::FrenetObserver::MinS", "", 0.0);
      max_s_ = params->GetReal("ML::FrenetObserver::MaxS", "", 100.0);
      min_d_ = params->GetReal("ML::FrenetObserver::MinD", "", -10.0);
      max_d_ = params->GetReal("ML::FrenetObserver::MaxD", "", 10.0);
      min_theta_ = params->GetReal("ML::FrenetObserver::MinTheta", "", -B_PI);
      max_theta_ = params->GetReal("ML::FrenetObserver::MaxTheta", "", B_PI);
      use_rel_s_values_ = params->GetBool(
        "ML::FrenetObserver::RelativeLonValues", "", true);
      observation_len_ = nearest_agent_num_ * 4 + 5;
  }

  double Norm(const double val, const double mi, const double ma) const {
    LOG_IF_EVERY_N(WARNING, (val < mi), 100) << "Val=" << val <<
      " < Lower Limit=" << mi;
    LOG_IF_EVERY_N(WARNING, (val > ma), 100) << "Val=" << val <<
      " > Upper Limit=" << ma;
    // Normalize to be within [-1, 1]
    return (val - (mi+ma)/2.0)/((ma-mi)/2.0);
  }

  FrenetState GetEgoFrenet(const ObservedWorld& observed_world) const {
    const auto ego_state = observed_world.CurrentEgoState();
    const auto ego_pos = observed_world.CurrentEgoPosition();
    const auto ego_corridor = observed_world.GetEgoAgent()
      ->GetRoadCorridor()->GetNearestLaneCorridor(ego_pos);
    FrenetState current_ego_frenet(ego_state, ego_corridor->GetCenterLine());
    return current_ego_frenet;
  }

  ObservedState GetEgoState(const ObservedWorld& observed_world) const {
    const auto current_ego_frenet = GetEgoFrenet(observed_world);
    ObservedState ego_nn_state(1, 5);
    const double normalized_angle = NormToPI(current_ego_frenet.angle);
    double ego_s = 0.0;
    if (use_rel_s_values_) {
      ego_s -= current_ego_frenet.lon;
    }
    ego_nn_state << Norm(current_ego_frenet.lon + ego_s, min_s_, max_s_),
                    Norm(current_ego_frenet.lat, min_d_, max_d_),
                    Norm(normalized_angle, min_theta_, max_theta_),
                    Norm(current_ego_frenet.vlon, min_vel_lon_, max_vel_lon_),
                    Norm(current_ego_frenet.vlat, min_vel_lat_, max_vel_lat_);
    return ego_nn_state;
  }


  ObservedState GetOtherAgentState(
    const AgentId& agent_id, const ObservedWorld& observed_world) const {
    const auto ego_pos = observed_world.CurrentEgoPosition();
    const auto lane_corridor = observed_world.GetEgoAgent()
      ->GetRoadCorridor()->GetNearestLaneCorridor(ego_pos);
    const auto ego_state = observed_world.CurrentEgoState();
    FrenetState current_ego_frenet(
      ego_state, lane_corridor->GetCenterLine());
    const auto ego_shape = observed_world.GetEgoAgent()->GetShape();
    const auto& other_state = observed_world.GetAgent(
      agent_id)->GetCurrentState();
    FrenetState other_frenet(
      other_state, lane_corridor->GetCenterLine());
    const auto other_shape = observed_world.GetAgent(agent_id)->GetShape();
    FrenetStateDifference state_diff(
      current_ego_frenet, ego_shape, other_frenet, other_shape);

    double ego_s = 0.0;
    if (use_rel_s_values_) {
      const auto current_ego_frenet = GetEgoFrenet(observed_world);
      ego_s -= current_ego_frenet.lon;
    }
    ObservedState other_nn_state(1, 4);
    other_nn_state <<
      Norm(state_diff.lon_zeroed ? 0.0 : state_diff.lon + ego_s, min_s_, max_s_),
      Norm(state_diff.lat_zeroed ? 0.0 : state_diff.lat, min_d_, max_d_),
      Norm(state_diff.vlon, min_vel_lon_, max_vel_lon_),
      Norm(state_diff.vlat, min_vel_lat_, max_vel_lat_);
    return other_nn_state;
  }

  ObservedState Observe(const ObservedWorld& observed_world) const {
    // find near agents (n)
    AgentMap nearest_agents = observed_world.GetNearestAgents(
      observed_world.CurrentEgoPosition(), nearest_agent_num_ + 1);

    // Build state
    int state_start_idx = 0;
    ObservedState state(1, observation_len_);
    state.setZero();

    // add ego agent state
    ObservedState obs_ego_agent_state = GetEgoState(observed_world);
    state.block(
      0, state_start_idx, 1, obs_ego_agent_state.cols()) = obs_ego_agent_state;
    state_start_idx = obs_ego_agent_state.cols();

    // add other states
    for (const auto& agent : nearest_agents) {
      if(agent.first == observed_world.GetEgoAgentId()) continue;

      const auto& agent_state = agent.second->GetCurrentPosition();
      double distance = Distance(
        observed_world.CurrentEgoPosition(), agent_state);
      if (distance > max_dist_) continue;

      ObservedState other_agent_state = GetOtherAgentState(
        agent.second->GetAgentId(), observed_world);
      state.block(
        0, state_start_idx, 1, other_agent_state.cols()) = other_agent_state;
      state_start_idx += other_agent_state.cols();
    }
    std::cout << state << std::endl;
    return state;
  }

  WorldPtr Reset(const WorldPtr& world) {
    return world;
  }

  Box<double> ObservationSpace() const {
    Matrix_t<double> low(1, observation_len_);
    low.setOnes();
    low = -1*low;
    Matrix_t<double> high(1, observation_len_);
    high.setOnes();
    std::tuple<int> shape{observation_len_};
    return Box<double>(low, high, shape);
  }

 private:
  int nearest_agent_num_, observation_len_;
  double min_theta_, max_theta_, min_vel_lon_,
         max_vel_lon_, min_vel_lat_,
         max_vel_lat_, max_dist_,
         min_d_, max_d_, min_s_, max_s_;
  bool use_rel_s_values_;
};

}  // namespace observers
}  // namespace bark_ml

#endif  // BARK_ML_OBSERVERS_FRENET_OBSERVER_HPP_