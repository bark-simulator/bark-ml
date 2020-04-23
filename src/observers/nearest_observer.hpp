// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
#define SRC_OBSERVERS_NEAREST_OBSERVER_HPP_

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
#include "src/commons/spaces.hpp"
#include "src/commons/commons.hpp"

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
using ObservedState = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using modules::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;


class NearestObserver {
 public:
  explicit NearestObserver(const ParamsPtr& params) :
    params_(params),
    state_size_(4) {
      nearest_agent_num_ =
        params_->GetInt(
          "ML::Observer::n_nearest_agents", "Nearest agents number", 4);
      min_lon_ = params_->GetReal("ML::Observer::min_lon", "", -200.);
      max_lon_ = params_->GetReal("ML::Observer::max_lon", "", 200.);
      min_lat_ = params_->GetReal("ML::Observer::min_lat", "", -10.);
      max_lat_ = params_->GetReal("ML::Observer::max_lat", "", 10.);
      min_theta_ = params_->GetReal("ML::Observer::min_theta", "", -3.14);
      max_theta_ = params_->GetReal("ML::Observer::max_theta", "", 3.14);
      min_vel_ = params_->GetReal("ML::Observer::min_vel", "", 0.0);
      max_vel_ = params_->GetReal("ML::Observer::max_vel", "", 25.0);
      max_dist_ = params_->GetReal("ML::Observer::max_dist", "", 75.0);
      observation_len_ = nearest_agent_num_ * state_size_;
  }

  ObservedState TransformState(
    const State& state,
    const Line& center_line,
    const float ref_lon = 0.0) const {
    Point2d pose(
      state(StateDefinition::X_POSITION),
      state(StateDefinition::Y_POSITION));
    FrenetPosition frenet_pos(pose, center_line);
    ObservedState ret_state(1, state_size_);
    ret_state <<
      Norm<float>(frenet_pos.lon - ref_lon, min_lon_, max_lon_),
      Norm<float>(frenet_pos.lat, min_lat_, max_lat_),
      Norm<float>(
        state(StateDefinition::THETA_POSITION), min_theta_, max_theta_),
      Norm<float>(state(StateDefinition::VEL_POSITION), min_vel_, max_vel_);
    return ret_state;
  }

  ObservedState Observe(const ObservedWorldPtr& world) const {
    int row_idx = 0;
    ObservedState state(1, observation_len_);
    state.setZero();
    // TODO(@hart): this should later be removed
    world->UpdateAgentRTree();

    // check in which lane corridor the goal is
    std::shared_ptr<const Agent> ego_agent = world->GetEgoAgent();
    BARK_EXPECT_TRUE(ego_agent != nullptr);
    const Point2d ego_pos = ego_agent->GetCurrentPosition();
    const auto& road_corridor = ego_agent->GetRoadCorridor();
    BARK_EXPECT_TRUE(road_corridor != nullptr);
    const auto& ego_goal =
      std::dynamic_pointer_cast<GoalDefinitionStateLimitsFrenet>(
        ego_agent->GetGoalDefinition());
    const auto& ego_goal_center_line = ego_goal->GetCenterLine();
    const auto& lane_corridors = road_corridor->GetUniqueLaneCorridors();
    auto ego_lane_corridor = lane_corridors[0];
    for (auto& lane_corr : lane_corridors) {
      if (Collide(lane_corr->GetMergedPolygon(), ego_goal_center_line)) {
        ego_lane_corridor = lane_corr;
      }
    }

    // find near agents (n)
    AgentMap nearest_agents = world->GetNearestAgents(
      ego_pos, nearest_agent_num_);

    // sort agents by distance and remove far away ones
    std::map<float, AgentPtr, std::greater<float>> distance_agent_map;
    for (const auto& agent : nearest_agents) {
      const auto& agent_state = agent.second->GetCurrentPosition();
      float distance = Distance(ego_pos, agent_state);
      if (distance < max_dist_)
        distance_agent_map[distance] = agent.second;
    }

    // transform ego agent state
    ObservedState obs_ego_agent_state =
      TransformState(ego_agent->GetCurrentState(),
                     ego_lane_corridor->GetCenterLine());
    state.block(0, row_idx*state_size_, 1, state_size_) = obs_ego_agent_state;
    row_idx++;

    // transform other states (lon. relative to ego agent's lon.)
    for (const auto& agent : distance_agent_map) {
      if (agent.second->GetAgentId() != ego_agent->GetAgentId()) {
        ObservedState other_agent_state =
          TransformState(agent.second->GetCurrentState(),
                         ego_lane_corridor->GetCenterLine(),
                         obs_ego_agent_state(0, 0));
        state.block(0, row_idx*state_size_, 1, state_size_) = other_agent_state;
        row_idx++;
      }
    }
    // state(0, 0) = Norm<float>(0., min_lon_, max_lon_);
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
  float min_lon_, max_lon_, min_lat_, max_lat_,
        min_theta_, max_theta_, min_vel_, max_vel_,
        max_dist_;
};

}  // namespace observers

#endif  // SRC_OBSERVERS_NEAREST_OBSERVER_HPP_
