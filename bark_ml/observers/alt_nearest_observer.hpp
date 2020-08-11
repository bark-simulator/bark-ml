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
#include <boost/geometry.hpp>

#include "bark/commons/params/params.hpp"
#include "bark/world/world.hpp"
#include "bark/world/observed_world.hpp"
#include "bark/world/goal_definition/goal_definition_state_limits_frenet.hpp"
#include "bark/models/dynamic/dynamic_model.hpp"
#include "src/commons/spaces.hpp"
#include "src/commons/commons.hpp"

namespace observers {
using bark::commons::ParamsPtr;
using bark::world::Agent;
using spaces::Box;
using commons::Norm;
using spaces::Matrix_t;
using bark::world::AgentMap;
using bark::world::AgentPtr;
using bark::world::WorldPtr;
using bark::world::goal_definition::GoalDefinitionStateLimitsFrenet;
using bark::world::ObservedWorldPtr;
using bark::geometry::Point2d;
using bark::geometry::Line;
using bark::geometry::Distance;
using bark::models::dynamic::StateDefinition;
using ObservedState = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using bark::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;

#define terminal_output_enabled false

class NearestObserver {
  public:
    explicit NearestObserver(const ParamsPtr& params) :
      params_(params),
      world_x_min_ (0.0),
      world_x_max_ (100.0),
      world_y_min_ (0.0),
      world_y_max_ (100.0),
      state_size_(4) {  //state size 4 fixed (X_Pos, Y_Pos, Theta, Vel)
        nearest_agent_num_ = params_->GetInt("ML::Observer::n_nearest_agents", "Nearest agents number", 4);
        min_theta_ = params_->GetReal("ML::Observer::min_theta", "", -3.14159);  //[rad]
        max_theta_ = params_->GetReal("ML::Observer::max_theta", "", 3.14159);   //[rad]
        min_vel_ = params_->GetReal("ML::Observer::min_vel", "", 0.0);  //[m/s]
        max_vel_ = params_->GetReal("ML::Observer::max_vel", "", 30.0); //[m/s]
        max_dist_ = params_->GetReal("ML::Observer::max_dist", "", 50.0); //[m]
        normalization_enabled = params_->GetBool("ML::Observer::normalization_enabled", "", true);
        distance_method_ = params_->GetInt("ML::Observer::distance_method", "", 2); //1=L1; 2=L2(default)
        observation_len_ = nearest_agent_num_ * state_size_;     
      }

  ObservedState NormalizeState(const State& state) const{    
    ObservedState ret_state(1, state_size_);
    if (normalization_enabled == true){
      ret_state <<
      Norm<double>(state(StateDefinition::X_POSITION), world_x_min_, world_x_max_),
      Norm<double>(state(StateDefinition::Y_POSITION), world_y_min_, world_y_max_),
      Norm<double>(state(StateDefinition::THETA_POSITION), min_theta_, max_theta_),
      Norm<double>(state(StateDefinition::VEL_POSITION), min_vel_, max_vel_);
      return ret_state;      
    }
    else{
      ret_state <<
      state(StateDefinition::X_POSITION),
      state(StateDefinition::Y_POSITION),
      state(StateDefinition::THETA_POSITION),
      state(StateDefinition::VEL_POSITION);
      return ret_state;
    }   
  }

  ObservedState observe(const ObservedWorldPtr& world) const {
    ObservedState state(1, observation_len_);
    state.setZero();
    
    // find near agents (n)
    std::shared_ptr<const Agent> ego_agent = world->GetEgoAgent();
    BARK_EXPECT_TRUE(ego_agent != nullptr);
    const Point2d ego_pos = ego_agent->GetCurrentPosition();
    AgentMap nearest_agents = world->GetNearestAgents(ego_pos, nearest_agent_num_);

    // sort agents by distance 
    std::map<float, AgentPtr, std::less<float>> distance_agent_map;
    for (const auto& agent : nearest_agents) {
      const auto& agent_state = agent.second->GetCurrentPosition();
      float distance = 0; //init  
      if (distance_method_ == 1){
        distance = L1_Distance(ego_pos, agent_state); //uses L1 Distance 
      }
      else{
        distance = Distance(ego_pos, agent_state); //uses L2 Distance
      }         
      if (distance < max_dist_) {   //remove far agents
        distance_agent_map[distance] = agent.second;        
        #if terminal_output_enabled==true
          auto view_state = agent.second->GetCurrentState();
          std::cout<<"agent Id: "<< agent.first << std::endl;
          std::cout<<"distance to ego_agent: "<< distance << std::endl;
          std::cout<<"State : \n"<< view_state << std::endl;
        #endif  
      }    
    }

    // transform ego agent state
    int col_idx = 0;
    ObservedState obs_ego_agent_state = NormalizeState(ego_agent->GetCurrentState());
    //insert normaized ego state at first postion in concatenated array
    state.block(0, col_idx*state_size_, 1, state_size_) = obs_ego_agent_state;
    col_idx++;

    #if terminal_output_enabled==true
      std::cout<<"ego_id: " << ego_agent->GetAgentId() << std::endl;
      std::cout<<"ego_state_normalized: \n" << obs_ego_agent_state << std::endl;
      std::cout<<"state_vector: \n" << state << std::endl;
    #endif

    // loop map of other agents (sorted by distance) 
    for (auto& agent : distance_agent_map) {
      if (agent.second->GetAgentId() != ego_agent->GetAgentId()) {
        ObservedState other_agent_state = NormalizeState(agent.second->GetCurrentState());
        state.block(0, col_idx*state_size_, 1, state_size_) = other_agent_state;
        col_idx++;

        #if terminal_output_enabled==true
          std::cout<<"agent_id: "<<agent.second->GetAgentId()<<std::endl;
          std::cout<<"agent_state_normalized: \n" << other_agent_state << std::endl;
          std::cout<<"state_vector: \n" << state << std::endl;
        #endif
      }
    }
    return state;
  }

  float L1_Distance (const Point2d &p1, const Point2d &p2) const {
    float dx = boost::geometry::get<0>(p1) - boost::geometry::get<0>(p2);
    float dy = boost::geometry::get<1>(p1) - boost::geometry::get<1>(p2);
    return (abs(dx) + abs(dy));
  }

  WorldPtr Reset(const WorldPtr& world, const std::vector<int>& agent_ids) {
    const auto& x_y = world->BoundingBox();
    Point2d bb0 = x_y.first;
    Point2d bb1 = x_y.second;
    world_x_min_ = bb0.get<0>();
    world_x_max_ = bb1.get<0>();
    world_y_min_ = bb0.get<1>();
    world_y_max_ = bb1.get<1>();
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
    bool normalization_enabled;
    const int state_size_;
    int nearest_agent_num_;
    int observation_len_;
    int distance_method_;
    float min_theta_, max_theta_, min_vel_, max_vel_, max_dist_;
    float world_x_min_, world_x_max_, world_y_min_, world_y_max_;
};

}  // namespace observers

#undef terminal_output_enabled
#endif  // SRC_OBSERVERS_NEAREST_OBSERVER_HPP_