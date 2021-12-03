// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_OBSERVERS_STATIC_OBSERVER_HPP_
#define BARK_ML_OBSERVERS_STATIC_OBSERVER_HPP_

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


class StaticObserver {
 public:
  explicit StaticObserver(const ParamsPtr& params) {
    min_vel_lon_ = params->GetReal("ML::StaticObserver::MinVelLon", "", -20.0);
    max_vel_lon_ = params->GetReal("ML::StaticObserver::MaxVelLon", "", 20.0);
    min_vel_lat_ = params->GetReal("ML::StaticObserver::MinVelLat", "", -10.0);
    max_vel_lat_ = params->GetReal("ML::StaticObserver::MaxVelLat", "", 10.0);
    max_dist_ = params->GetReal("ML::StaticObserver::MaxDist", "", 75.0);
    min_s_ = params->GetReal("ML::StaticObserver::MinS", "", 0.0);
    max_s_ = params->GetReal("ML::StaticObserver::MaxS", "", 100.0);
    min_d_ = params->GetReal("ML::StaticObserver::MinD", "", -10.0);
    max_d_ = params->GetReal("ML::StaticObserver::MaxD", "", 10.0);
    min_theta_ = params->GetReal("ML::StaticObserver::MinTheta", "", -B_PI);
    max_theta_ = params->GetReal("ML::StaticObserver::MaxTheta", "", B_PI);
    min_steering_rate_ = params->GetReal(
      "ML::StaticObserver::MinSteeringRate", "", -4.0);;
    max_steering_rate_ = params->GetReal(
      "ML::StaticObserver::MaxSteeringRate", "", 4.0);;
    observation_len_ = 5+4;
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
  /**
   * @brief  Returns the normalized ego state
   * @note   [lat, angle, v_lon, v_laat]
   * @param  observed_world: Observed World
   * @retval Normalized ego state.
   */
  ObservedState GetEgoState(const ObservedWorld& observed_world) const {
    const auto current_ego_frenet = GetEgoFrenet(observed_world);
    ObservedState ego_nn_state(1, 5);
    const double normalized_angle = NormToPI(current_ego_frenet.angle);
    // TODO: check which dynamic model is used and integrate all the states
    //       depending on that
    auto ego_agent = observed_world.GetEgoAgent();
    auto state = ego_agent->GetCurrentState();
    ego_nn_state << Norm(current_ego_frenet.lat, min_d_, max_d_),
                    Norm(normalized_angle, min_theta_, max_theta_),
                    Norm(current_ego_frenet.vlon, min_vel_lon_, max_vel_lon_),
                    Norm(current_ego_frenet.vlat, min_vel_lat_, max_vel_lat_),
                    Norm(state[5], min_steering_rate_, max_steering_rate_);
    return ego_nn_state;
  }

  /**
   * @brief  Returns the difference between two FrenetStates (other and ego)
   * @note
   * @param  agent_id: Agent-ID
   * @param  observed_world: Observed World
   * @retval FrenetStateDifference state
   */
  FrenetStateDifference GetOtherAgentState(
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

    return state_diff;
  }

  ObservedState Observe(const ObservedWorld& observed_world) const {
    // find near agents (n)
    AgentMap nearest_agents = observed_world.GetNearestAgents(
      observed_world.CurrentEgoPosition(), 10);

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
    double min_lat_dist_front_left = 10000.0, min_lat_dist_front_right = -10000.0,
           min_lon_dist_front_left = 10000.0, min_lon_dist_front_right = 10000.0;
    bool exists_front_left = false, exists_front_right = false;

    for (const auto& agent : nearest_agents) {
      if(agent.first == observed_world.GetEgoAgentId()) continue;

      const auto& agent_state = agent.second->GetCurrentPosition();
      double distance = Distance(
        observed_world.CurrentEgoPosition(), agent_state);
      if (distance > max_dist_) continue;

      auto state_diff = GetOtherAgentState(agent.second->GetAgentId(), observed_world);

      // Objects behind ego are not considered
      double lon = state_diff.lon_zeroed ? 0.0 : state_diff.lon;
      double lat = state_diff.lat_zeroed ? 0.0 : state_diff.lat;
      if (lon < 0) continue;
      if(lon < min_lon_dist_front_left && state_diff.lat >= 0) {
        min_lon_dist_front_left = lon;
        exists_front_left = true;
      }
      if(lon < min_lon_dist_front_right && state_diff.lat < 0) {
        min_lon_dist_front_right = state_diff.lon_zeroed ? 0.0 : state_diff.lon;
        exists_front_right = true;
      }
      if(state_diff.lat >=0 && lat < min_lat_dist_front_left) {
        min_lat_dist_front_left = lat;
      }
      if(state_diff.lat < 0 && lat > min_lat_dist_front_right) {
        min_lat_dist_front_right = lat;
      }
    }
    state(0, state_start_idx) = exists_front_left ? Norm(std::abs(min_lon_dist_front_left), min_s_, max_s_) : -1.0;
    state(0, state_start_idx + 1) = exists_front_left ?  Norm(std::abs(min_lat_dist_front_left), min_d_, max_d_) : -1.0;
    state(0, state_start_idx + 2) = exists_front_right ? Norm(std::abs(min_lon_dist_front_right), min_s_, max_s_) : -1.0;
    state(0, state_start_idx + 3) = exists_front_right ? Norm(std::abs(min_lat_dist_front_right), min_d_, max_d_) : -1.0;
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
  int observation_len_;
  double min_theta_, max_theta_, min_vel_lon_,
         max_vel_lon_, min_vel_lat_,
         max_vel_lat_, max_dist_,
         min_d_, max_d_, min_s_, max_s_,
         min_steering_rate_, max_steering_rate_;
};

}  // namespace observers
}  // namespace bark_ml

#endif  // BARK_ML_OBSERVERS_STATIC_OBSERVER_HPP_