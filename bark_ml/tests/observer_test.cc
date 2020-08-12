// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#include "gtest/gtest.h" 
#include "bark/commons/params/params.hpp"
#include "bark/geometry/geometry.hpp"
#include "bark/commons/params/setter_params.hpp"
#include "bark_ml/observers/alt_nearest_observer.hpp"
#include "bark/world/tests/make_test_world.hpp"


#include "bark/models/behavior/constant_acceleration/constant_acceleration.hpp"
#include "bark/models/behavior/idm/idm_classic.hpp"
#include "bark/models/dynamic/single_track.hpp"
#include "bark/models/execution/interpolation/interpolate.hpp"

using namespace bark::models::dynamic;
using namespace bark::models::execution;
using namespace bark::commons;
using namespace bark::models::behavior;
using namespace bark::world::map;
using namespace bark::world;
using namespace bark::geometry;
using namespace bark::world::tests;

// observer
using bark_ml::observers::NearestObserver;
using bark_ml::observers::ObservedState;
using bark_ml::commons::Norm;

//function: returns pointer to default params

TEST(observers, test_state_vector_length){
  //auto params = std::make_shared<DefaultParams>();
  auto params = std::make_shared<SetterParams>();
  const int state_size_ = 4;  
  int nearest_agent_num_ = 3;
  params->SetInt("ML::Observer::n_nearest_agents", nearest_agent_num_);  
  int res_rows;
  int res_length; 

  
  ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
  DynamicModelPtr dyn_model(new SingleTrackModel(params));
  BehaviorModelPtr beh_model_const(new BehaviorConstantAcceleration(params));
  BehaviorModelPtr beh_model_idm(new BehaviorIDMClassic(params));
  Polygon polygon(Pose(1.25, 1, 0), std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(4, 2), Point2d(4, 0), Point2d(0, 0)});


  /*test1: add only one agent; nearest_agent_num_=4*/ 
  State init_state0(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  init_state0 << 0.0, 10, 10, 0, 5.0;
  AgentPtr agent0(new Agent(init_state0, beh_model_idm, dyn_model, exec_model, polygon, params)); //ego
  WorldPtr world(new World(params));
  world->AddAgent(agent0);
  world->UpdateAgentRTree();
  WorldPtr world1 = world->Clone();

  ObservedWorld observed_world1(world1, world1->GetAgents().begin()->second->GetAgentId());
  ObservedWorldPtr obs_world_ptr1 = std::make_shared<ObservedWorld>(observed_world1);
  
  //create instance of Observer and pass observed world
  NearestObserver TestObserver1(params);
  ObservedState res = TestObserver1.Observe(obs_world_ptr1);
  //std::cout << res << std::endl;
  
  res_rows = res.rows();
  res_length = res.cols();
  
  EXPECT_EQ(res_rows, 1);
  EXPECT_EQ(res_length, nearest_agent_num_ * state_size_);
  //std::vector<int> agent_ids1{0};
  //TestObserver1.Reset(world);  
}

TEST(observers, test_state_vector_length2){
  auto params = std::make_shared<SetterParams>();
  const int state_size_ = 4;  
  int nearest_agent_num_ = 4;
  params->SetInt("ML::Observer::n_nearest_agents", nearest_agent_num_);  
  int res_rows;
  int res_length; 

  ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
  DynamicModelPtr dyn_model(new SingleTrackModel(params));
  BehaviorModelPtr beh_model_const(new BehaviorConstantAcceleration(params));
  BehaviorModelPtr beh_model_idm(new BehaviorIDMClassic(params));
  Polygon polygon(Pose(1.25, 1, 0), std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(4, 2), Point2d(4, 0), Point2d(0, 0)});

  /*test2: add more than 4 agents; nearest_agent_num_=4*/ 
  State init_state0(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state1(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state2(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state3(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state4(static_cast<int>(StateDefinition::MIN_STATE_SIZE));

  init_state0 << 0.0, 10, 10, 0, 5.0; //all agents close to ego
  init_state1 << 0.0, 11, 12, 0, 5.0;
  init_state2 << 0.0, -11, -12, 0, 5;       
  init_state3 << 0.0, 12, 10, 0.0, 5.0; 
  init_state4 << 0.0, 10, 11, 0.0, 5.0;   

  AgentPtr agent0(new Agent(init_state0, beh_model_idm, dyn_model, exec_model, polygon, params)); //ego
  AgentPtr agent1(new Agent(init_state1, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent2(new Agent(init_state2, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent3(new Agent(init_state3, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent4(new Agent(init_state4, beh_model_const, dyn_model, exec_model, polygon, params));

  WorldPtr world(new World(params));
  world->AddAgent(agent0);
  world->AddAgent(agent1);
  world->AddAgent(agent2);
  world->AddAgent(agent3);
  world->AddAgent(agent4);
  world->UpdateAgentRTree();
  WorldPtr world1 = world->Clone();

  ObservedWorld observed_world(world1, world1->GetAgents().begin()->second->GetAgentId());
  ObservedWorldPtr obs_world_ptr = std::make_shared<ObservedWorld>(observed_world);
  
  //create instance of Observer and pass observed world
  NearestObserver TestObserver2(params);
  ObservedState res = TestObserver2.Observe(obs_world_ptr);
  //std::cout << res << std::endl;
  
  res_rows = res.rows();
  res_length = res.cols();
  
  EXPECT_EQ(res_rows, 1);
  EXPECT_EQ(res_length, nearest_agent_num_ * state_size_);
  //std::vector<int> agent_ids1{0};
  //TestObserver2.Reset(world);
}

TEST(observers, test_max_distance){
  //auto params = std::make_shared<DefaultParams>();
  auto params = std::make_shared<SetterParams>();
  const int state_size_ = 4;  
  int max_dist_ = 75.0;
  params->SetInt("ML::Observer::n_nearest_agents", 4); 
  params->SetReal("ML::Observer::max_dist", max_dist_);
  params->SetInt("ML::Observer::distance_method", 2); //L2 Distance
  params->SetBool("ML::Observer::normalization_enabled", false); //switch off normalization to avoid implicit testing  

  //float max_dist_ = params->GetReal("ML::Observer::max_dist", "", 75); //[m]
  //int distance_method_ = params->GetInt("ML::Observer::distance_method", "Nearest agents number", 2); //1=L1; 2=L2(default)

  ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
  DynamicModelPtr dyn_model(new SingleTrackModel(params));
  BehaviorModelPtr beh_model_const(new BehaviorConstantAcceleration(params));
  BehaviorModelPtr beh_model_idm(new BehaviorIDMClassic(params));
  Polygon polygon(Pose(1.25, 1, 0), std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(4, 2), Point2d(4, 0), Point2d(0, 0)});

  /*test2: add more than 4 agents; nearest_agent_num_=4*/ 
  State init_state0(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state1(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state2(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state3(static_cast<int>(StateDefinition::MIN_STATE_SIZE));  

  init_state0 << 0.0, 0, 0, 0, 5.0; //all agents close to ego
  init_state1 << 0.0, -54, -54, 0, 5.0; //L2=76.34 --> too far   
  init_state2 << 0.0, 20, 73, 0.0, 5.0; //L2=75.69 --> too far
  init_state3 << 0.0, 53, 53, 0, 5.0; //L2 = 74.95 
  
  AgentPtr agent0(new Agent(init_state0, beh_model_idm, dyn_model, exec_model, polygon, params)); //ego
  AgentPtr agent1(new Agent(init_state1, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent2(new Agent(init_state2, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent3(new Agent(init_state3, beh_model_const, dyn_model, exec_model, polygon, params));
  
  WorldPtr world(new World(params));
  world->AddAgent(agent0);
  world->AddAgent(agent1);
  world->AddAgent(agent2);
  world->AddAgent(agent3);
  world->UpdateAgentRTree();
  WorldPtr world1 = world->Clone();

  ObservedWorld observed_world(world1, world1->GetAgents().begin()->second->GetAgentId());
  ObservedWorldPtr obs_world_ptr = std::make_shared<ObservedWorld>(observed_world);
  
  //create instance of Observer and pass observed world
  NearestObserver TestObserver3(params);
  ObservedState res = TestObserver3.Observe(obs_world_ptr);
  //std::cout << res << std::endl;
  
  float obs_X_pos_agent3 = res.coeff(0,4);
  float obs_Y_pos_agent3 = res.coeff(0,5);
  
  float max_devation = 0.00005;

  EXPECT_NEAR(obs_X_pos_agent3, init_state3(StateDefinition::X_POSITION), max_devation); //53m normalized
  EXPECT_NEAR(obs_Y_pos_agent3, init_state3(StateDefinition::Y_POSITION), max_devation);
  for(int zwerg=8; zwerg<16; zwerg++){
    int pos = res.coeff(0,zwerg);
    EXPECT_EQ(pos, 0);
  };
  //std::vector<int> agent_ids1{0};
  //TestObserver3.Reset(world);
}

TEST(observers, test_state_vector_order){
  //auto params = std::make_shared<DefaultParams>();
  auto params = std::make_shared<SetterParams>();
  const int state_size_ = 4;  
  params->SetInt("ML::Observer::n_nearest_agents", 4);
  params->SetReal("ML::Observer::max_dist", 100);
  params->SetInt("ML::Observer::distance_method", 2); //L2 Distance
  params->SetBool("ML::Observer::normalization_enabled", false); //switch off normalization to avoid implicit testing   
  
  ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
  DynamicModelPtr dyn_model(new SingleTrackModel(params));
  BehaviorModelPtr beh_model_const(new BehaviorConstantAcceleration(params));
  BehaviorModelPtr beh_model_idm(new BehaviorIDMClassic(params));
  Polygon polygon(Pose(1.25, 1, 0), std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(4, 2), Point2d(4, 0), Point2d(0, 0)});

  /*test2: add more than 4 agents; nearest_agent_num_=4*/ 
  State init_state0(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state1(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state2(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state3(static_cast<int>(StateDefinition::MIN_STATE_SIZE));  

  init_state0 << 0.0, 0, 0, 0, 5.0;       //ego
  init_state1 << 0.0, -15, -15, 0, 5.0;   //3rd closest
  init_state2 << 0.0, 10, 0, 0.0, 5.0;    //1st closest
  init_state3 << 0.0, 15, -5, 0.0, 10.0;  //2nd closest
  
  AgentPtr agent0(new Agent(init_state0, beh_model_idm, dyn_model, exec_model, polygon, params)); //ego
  AgentPtr agent1(new Agent(init_state1, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent2(new Agent(init_state2, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent3(new Agent(init_state3, beh_model_const, dyn_model, exec_model, polygon, params));
  
  WorldPtr world(new World(params));
  world->AddAgent(agent3);  //add in different order
  world->AddAgent(agent2);
  world->AddAgent(agent1);
  world->AddAgent(agent0);
  world->UpdateAgentRTree();
  WorldPtr world1 = world->Clone();

  ObservedWorld observed_world(world1, world1->GetAgents().begin()->second->GetAgentId());
  ObservedWorldPtr obs_world_ptr = std::make_shared<ObservedWorld>(observed_world);
  
  //create instance of Observer and pass observed world
  NearestObserver TestObserver4(params);
  ObservedState res = TestObserver4.Observe(obs_world_ptr);
  //std::cout << res << std::endl;

  //select here psoitions in the order how it should be according the state definition
  float obs_X_pos_agent0 = res.coeff(0,0);
  float obs_Y_pos_agent0 = res.coeff(0,1);
  float obs_X_pos_agent1 = res.coeff(0,12);
  float obs_Y_pos_agent1 = res.coeff(0,13);
  float obs_X_pos_agent2 = res.coeff(0,4);
  float obs_Y_pos_agent2 = res.coeff(0,5);
  float obs_X_pos_agent3 = res.coeff(0,8);
  float obs_Y_pos_agent3 = res.coeff(0,9);  
  
  float max_devation = 0.00005;

  EXPECT_NEAR(obs_X_pos_agent0, init_state0(StateDefinition::X_POSITION), max_devation);
  EXPECT_NEAR(obs_Y_pos_agent0, init_state0(StateDefinition::Y_POSITION), max_devation);
  EXPECT_NEAR(obs_X_pos_agent1, init_state1(StateDefinition::X_POSITION), max_devation);
  EXPECT_NEAR(obs_Y_pos_agent1, init_state1(StateDefinition::Y_POSITION), max_devation);
  EXPECT_NEAR(obs_X_pos_agent2, init_state2(StateDefinition::X_POSITION), max_devation);
  EXPECT_NEAR(obs_Y_pos_agent2, init_state2(StateDefinition::Y_POSITION), max_devation);
  EXPECT_NEAR(obs_X_pos_agent3, init_state3(StateDefinition::X_POSITION), max_devation);
  EXPECT_NEAR(obs_Y_pos_agent3, init_state3(StateDefinition::Y_POSITION), max_devation);
     
  //std::vector<int> agent_ids1{0};
  //TestObserver4.Reset(world);
}

TEST(observers, test_normalization){
  //auto params = std::make_shared<DefaultParams>();
  auto params = std::make_shared<SetterParams>();
  const int state_size_ = 4;  
  params->SetInt("ML::Observer::n_nearest_agents", 4);
  params->SetReal("ML::Observer::max_dist", 100);
  params->SetInt("ML::Observer::distance_method", 2); //L2 Distance
  params->SetBool("ML::Observer::normalization_enabled", true); //switch on normalization 
  params->SetReal("ML::Observer::min_theta", -3.14159);  //[rad]
  params->SetReal("ML::Observer::max_theta", 3.14159);   //[rad]
  params->SetReal("ML::Observer::min_vel", 0.0);  //[m/s]
  params->SetReal("ML::Observer::max_vel", 25.0); //[m/s]
    
  float world_x_min_ = 0;
  float world_x_max_ = 100.0;
  float world_y_min_ = 0.0;
  float world_y_max_ = 100.0;
  
  //std::cout<<"num_agents: "<<nearest_agent_num_<<std::endl;

  ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
  DynamicModelPtr dyn_model(new SingleTrackModel(params));
  BehaviorModelPtr beh_model_const(new BehaviorConstantAcceleration(params));
  BehaviorModelPtr beh_model_idm(new BehaviorIDMClassic(params));
  Polygon polygon(Pose(1.25, 1, 0), std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(4, 2), Point2d(4, 0), Point2d(0, 0)});

  State init_state0(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state1(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  

  //initialize agent states:
  //Time[s], X_Pos[m], Y_Pos[m], Theata[rad], Vel[m/s]
  /*
  init_state0 << 0.0, 0.0, 0.0, 0.0, 5.0;
  init_state1 << 0.0, 10.0, 20.0, -1.5708, 5.0;
  init_state2 << 0.0, -10, 5, 1.5708, 10;       //90deg rotated, would crash ego agent
  init_state3 << 0.0, 0.02, -9999.95, 0.0, 5.0; //check computational limits
  init_state4 << 0.0, 10000, 10000, 0.0, 5.0;   //add to verify that oberver only uses defined maximum number of agents
  */

  init_state0 << 0.0, 10, -10, 0.1, 5.0;
  init_state1 << 0.0, 0, 20, -3.14, 20;
  
  //create agents
  AgentPtr agent0(new Agent(init_state0, beh_model_idm, dyn_model, exec_model, polygon, params)); //ego
  AgentPtr agent1(new Agent(init_state1, beh_model_const, dyn_model, exec_model, polygon, params));
 
  //create world
  WorldPtr world(new World(params));
  world->AddAgent(agent0);
  world->AddAgent(agent1);
  world->UpdateAgentRTree();
  WorldPtr world1 = world->Clone();

  ObservedWorld observed_world1(world1, world1->GetAgents().begin()->second->GetAgentId());
  ObservedWorldPtr obs_world_ptr1 = std::make_shared<ObservedWorld>(observed_world1); 

  //create instance of Observer and pass observed world
  NearestObserver TestObserver1(params);
  ObservedState res = TestObserver1.Observe(obs_world_ptr1);
  //std::cout << res << std::endl;

  float obs_X_pos_agent0 = res.coeff(0,0);
  float obs_Y_pos_agent0 = res.coeff(0,1);
  float obs_theta_agent0 = res.coeff(0,2);
  float obs_vel_agent0   = res.coeff(0,3);
  float obs_X_pos_agent1 = res.coeff(0,4);
  float obs_Y_pos_agent1 = res.coeff(0,5);
  float obs_theta_agent1 = res.coeff(0,6);
  float obs_vel_agent1   = res.coeff(0,7); 

  float max_devation = 0.00005;

  EXPECT_NEAR(obs_X_pos_agent0, 0.1, max_devation);
  EXPECT_NEAR(obs_Y_pos_agent0, -0.1, max_devation);
  EXPECT_NEAR(obs_theta_agent0, 0.5159155, max_devation);
  EXPECT_NEAR(obs_vel_agent0, 0.2, max_devation);
  EXPECT_NEAR(obs_X_pos_agent1, 0, max_devation);
  EXPECT_NEAR(obs_Y_pos_agent1, 0.2, max_devation);
  EXPECT_NEAR(obs_theta_agent1, 0.0002531, max_devation);
  EXPECT_NEAR(obs_vel_agent1, 0.8, max_devation);

  // Reset
  //std::vector<int> agent_ids1{0};
  //TestObserver1.Reset(world);
}
