// Copyright (c) 2020 fortiss GmbH
//
// Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "bark/world/observed_world.hpp"
#include "bark/commons/params/setter_params.hpp"
#include "bark/geometry/polygon.hpp"
#include "bark/geometry/standard_shapes.hpp"
#include "bark/models/behavior/constant_acceleration/constant_acceleration.hpp"
#include "bark/models/behavior/motion_primitives/continuous_actions.hpp"
#include "bark/models/dynamic/single_track.hpp"
#include "bark/models/execution/interpolation/interpolate.hpp"
#include "bark/world/evaluation/evaluator_collision_agents.hpp"
#include "bark/world/goal_definition/goal_definition.hpp"
#include "bark/world/goal_definition/goal_definition_polygon.hpp"
#include "bark/world/map/map_interface.hpp"
#include "bark/world/map/roadgraph.hpp"
#include "bark/world/objects/agent.hpp"
#include "bark/world/opendrive/opendrive.hpp"
#include "bark/world/tests/make_test_world.hpp"
#include "bark/world/tests/make_test_xodr_map.hpp"
#include "bark_ml/observers/nearest_observer.hpp"
#include "gtest/gtest.h"

using namespace bark::models::dynamic;
using namespace bark::models::behavior;
using namespace bark::models::execution;
using namespace bark::world::map;
using namespace bark_ml::observers;

using bark::commons::SetterParams;
using bark::commons::transformation::FrenetPosition;
using bark::commons::transformation::FrenetStateDifference;
using bark::geometry::Model3D;
using bark::geometry::Point2d;
using bark::geometry::Polygon;
using bark::geometry::Pose;
using bark::geometry::B_PI;
using bark::geometry::standard_shapes::CarRectangle;
using bark::geometry::standard_shapes::GenerateGoalRectangle;
using bark::world::FrontRearAgents;
using bark::world::ObservedWorld;
using bark::world::ObservedWorldPtr;
using bark::world::World;
using bark::world::WorldPtr;
using bark::world::goal_definition::GoalDefinitionPolygon;
using bark::world::objects::Agent;
using bark::world::objects::AgentPtr;
using bark::world::opendrive::OpenDriveMapPtr;
using bark::world::tests::MakeXodrMapOneRoadTwoLanes;

using StateDefinition::MIN_STATE_SIZE;

const double left_lane_y = -1.75;
const double right_lane_y = -5.25;

WorldPtr create_world() {
    auto params = std::make_shared<SetterParams>();
    WorldPtr world(new World(params));
    OpenDriveMapPtr open_drive_map = MakeXodrMapOneRoadTwoLanes();
    MapInterfacePtr map_interface = std::make_shared<MapInterface>();
    map_interface->interface_from_opendrive(open_drive_map);
    world->SetMap(map_interface);
    return world;
}

AgentId create_agent(WorldPtr world,
            double x, double y, double v, double theta) {
    auto params = std::make_shared<SetterParams>();
    // Goal Definition
    Polygon polygon = GenerateGoalRectangle(6,3);
    std::shared_ptr<Polygon> goal_polygon(
        std::dynamic_pointer_cast<Polygon>(polygon.Translate(Point2d(50, -2))));
    auto goal_ptr = std::make_shared<GoalDefinitionPolygon>(*goal_polygon);

    // Setting Up Agents (one in front of another)
    ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
    DynamicModelPtr dyn_model(new SingleTrackModel(params));
    BehaviorModelPtr beh_model(new BehaviorConstantAcceleration(params));
    Polygon car_polygon = CarRectangle();

    State init_state(static_cast<int>(MIN_STATE_SIZE));
    init_state << 0.0, x, y, theta, v;
    AgentPtr agent(new Agent(init_state, beh_model, dyn_model, exec_model,
                                car_polygon, params, goal_ptr, world->GetMap(),
                                Model3D())); 
    world->AddAgent(agent);
    return agent->GetAgentId();
}


TEST(nearest_observer, agent_front_same_lane ) { 
    auto params = std::make_shared<SetterParams>();
    params->SetInt("ML::NearestObserver::NNearestAgents", 2);
    params->SetReal("ML::NearestObserver::MinVelLon", -10.0);
    params->SetReal("ML::NearestObserver::MaxVelLon", 10.0);
    params->SetReal("ML::NearestObserver::MinVelLat", -10.0);
    params->SetReal("ML::NearestObserver::MaxVelLat", 10.0);
    params->SetReal("ML::NearestObserver::MaxDist", 5.0);
    params->SetReal("ML::NearestObserver::MinS", -20.0);
    params->SetReal("ML::NearestObserver::MaxS", 20.0);
    params->SetReal("ML::NearestObserver::MinD", -10.0);
    params->SetReal("ML::NearestObserver::MaxD", 10.0);
    params->SetReal("ML::NearestObserver::MinTheta", -B_PI);
    params->SetReal("ML::NearestObserver::MaxTheta", B_PI);
    NearestObserver nn_observer(params);

    const auto world = create_world();
    const auto id1 = create_agent(world, 5.0, left_lane_y, 3.5, 0.0);
    const auto id2 = create_agent(world, 10.0, left_lane_y, 3.6, 0.0);
    world->UpdateAgentRTree();
  
    const ObservedWorld observed_world(world, id1);

    const auto observed_nn_state = nn_observer.Observe(observed_world);
    EXPECT_EQ(observed_nn_state.cols(), 6+2*4);

    // Ego lon, lat, theta, vlon, vlat, lane id
    EXPECT_EQ(observed_nn_state(0, 0), 5.0/20.0);
    EXPECT_EQ(observed_nn_state(0, 1), 0.0);
    EXPECT_EQ(observed_nn_state(0, 2), 0.0);
    EXPECT_EQ(observed_nn_state(0, 3), 3.5/10.0);
    EXPECT_EQ(observed_nn_state(0, 4), 0.0);
    EXPECT_EQ(observed_nn_state(0, 5), 0.0);

    // Other dellon, dellat, delvlon, delvlat
    const auto es = observed_world.GetEgoAgent()->GetShape();
    EXPECT_EQ(observed_nn_state(0, 6), (10.0 - 5.0 - es.front_dist_ - es.rear_dist_)/20.0);
    EXPECT_EQ(observed_nn_state(0, 7), 0.0);
    EXPECT_NEAR(observed_nn_state(0, 8), 0.1/10.0, 0.0001);
    EXPECT_EQ(observed_nn_state(0, 9), 0.0);

    // Only one other agent in world, rest positions zeroed
    EXPECT_EQ(observed_nn_state(0, 10), 0.0);
    EXPECT_EQ(observed_nn_state(0, 11), 0.0);
    EXPECT_EQ(observed_nn_state(0, 12), 0.0);
    EXPECT_EQ(observed_nn_state(0, 13), 0.0);
}

TEST(nearest_observer, agent_front_other_lane_offsets) { 
    auto params = std::make_shared<SetterParams>();
    params->SetInt("ML::NearestObserver::NNearestAgents", 4);
    params->SetReal("ML::NearestObserver::MinVelLon", -10.0);
    params->SetReal("ML::NearestObserver::MaxVelLon", 10.0);
    params->SetReal("ML::NearestObserver::MinVelLat", -10.0);
    params->SetReal("ML::NearestObserver::MaxVelLat", 10.0);
    params->SetReal("ML::NearestObserver::MaxDist", 50.0);
    params->SetReal("ML::NearestObserver::MinS", -50.0);
    params->SetReal("ML::NearestObserver::MaxS", 50.0);
    params->SetReal("ML::NearestObserver::MinD", -15.0);
    params->SetReal("ML::NearestObserver::MaxD", 15.0);
    params->SetReal("ML::NearestObserver::MinTheta", -B_PI);
    params->SetReal("ML::NearestObserver::MaxTheta", B_PI);
    NearestObserver nn_observer(params);

    const auto world = create_world();
    const auto id1 = create_agent(world, 20.0, right_lane_y-0.5, 3.8, 0.4);
    const auto id2 = create_agent(world, 26.0, left_lane_y+1.0, 4.6, -0.6);
    const auto id3 = create_agent(world, -7.5, right_lane_y-0.6, 3.6, 0.0);
    const auto id4 = create_agent(world, 100.0, 0, 0, 0); // outside distance limit
    world->UpdateAgentRTree();
  
    const ObservedWorld observed_world(world, id1);

    const auto observed_nn_state = nn_observer.Observe(observed_world);
    EXPECT_EQ(observed_nn_state.cols(), 6+4*4);

    // Ego lon, lat, theta, vlon, vlat, lane id
    EXPECT_EQ(observed_nn_state(0, 0), 20.0/50.0);
    EXPECT_EQ(observed_nn_state(0, 1), -0.5/15.0);
    EXPECT_NEAR(observed_nn_state(0, 2), -0.4/B_PI, 0.0001);
    EXPECT_NEAR(observed_nn_state(0, 3), 3.8*cos(0.4)/10.0, 0.0001);
    EXPECT_NEAR(observed_nn_state(0, 4), 3.8*sin(0.4)/10.0, 0.0001);
    EXPECT_EQ(observed_nn_state(0, 5), 1.0);

    // Other dellon, dellat, delvlon, delvlat
    const auto es = observed_world.GetEgoAgent()->GetShape();
    EXPECT_NEAR(observed_nn_state(0, 6), (26.0 - 20.0 - es.front_dist_*cos(0.4) - es.rear_dist_*cos(0.6) -
                                             es.left_dist_*sin(0.4) - es.left_dist_*sin(0.6)  )/50.0, 0.0001);
    EXPECT_NEAR(observed_nn_state(0, 7), (left_lane_y +  1.0 - right_lane_y + 0.5 - es.front_dist_*sin(0.4)-
                                             es.front_dist_*sin(0.6) - es.left_dist_*cos(0.4) -
                                             es.left_dist_*cos(0.6)    )/15.0, 0.0001);
    EXPECT_NEAR(observed_nn_state(0, 8), (3.8*cos(0.4) - 4.6*cos(0.6) )/10.0, 0.0001);
    EXPECT_NEAR(observed_nn_state(0, 9), (3.8*sin(0.4) + 4.6*sin(0.6) )/10.0, 0.0001);

    // Only one other agent in world, rest positions zeroed
    EXPECT_EQ(observed_nn_state(0, 10), 0.0);
    EXPECT_EQ(observed_nn_state(0, 11), 0.0);
    EXPECT_EQ(observed_nn_state(0, 12), 0.0);
    EXPECT_EQ(observed_nn_state(0, 13), 0.0);
}

