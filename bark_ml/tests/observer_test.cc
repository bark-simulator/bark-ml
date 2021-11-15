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
#include "bark_ml/observers/static_observer.hpp"
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


TEST(static_observer, agents_exist ) { 
    auto params = std::make_shared<SetterParams>();
    params->SetReal("ML::StaticObserver::MinVelLon", -20.0);
    params->SetReal("ML::StaticObserver::MaxVelLon", 20.0);
    params->SetReal("ML::StaticObserver::MinVelLat", -10.0);
    params->SetReal("ML::StaticObserver::MaxVelLat", 10.0);
    params->SetReal("ML::StaticObserver::MaxDist", 100.0);
    params->SetReal("ML::StaticObserver::MinS", -20.0);
    params->SetReal("ML::StaticObserver::MaxS", 20.0);
    params->SetReal("ML::StaticObserver::MinD", -10.0);
    params->SetReal("ML::StaticObserver::MaxD", 10.0);
    params->SetReal("ML::StaticObserver::MinTheta", -B_PI);
    params->SetReal("ML::StaticObserver::MaxTheta", B_PI);
    StaticObserver nn_observer(params);

    const auto world = create_world();
    const auto id1 = create_agent(world, 5.0, left_lane_y, 3.5, 0.0);

    // before ego on the left 
    const auto id2 = create_agent(world, 10.0, left_lane_y + 4.0, 3.6, 0.0);

    // before ego on the left -> should not be considered
    const auto id3 = create_agent(world, 20.0, left_lane_y + 5.0, 3.6, 0.0);

    //on the side of ego on the right
    const auto id4 = create_agent(world, 6.0, left_lane_y - 3.0, 3.6, 0.0);

    //o before ego on the right -> should not be considered
    const auto id5 = create_agent(world, 15.0, left_lane_y - 6.0, 3.6, 0.0);

    // behind ego -> should not be considered
    const auto id6 = create_agent(world, 0.0, left_lane_y + 0.0, 3.6, 0.0);

    world->UpdateAgentRTree();
  
    const ObservedWorld observed_world(world, id1);

    const auto observed_nn_state = nn_observer.Observe(observed_world);
    EXPECT_EQ(observed_nn_state.cols(), 8);

    // Ego lat, theta, vlon, vlat
    EXPECT_EQ(observed_nn_state(0, 0), 0.0);
    EXPECT_EQ(observed_nn_state(0, 1), 0.0);
    EXPECT_NEAR(observed_nn_state(0, 2), 3.5/20.0, 0.001);
    EXPECT_EQ(observed_nn_state(0, 3), 0.0);

    const auto es = observed_world.GetEgoAgent()->GetShape();
    // Nearest lon/lat left
    EXPECT_NEAR(observed_nn_state(0, 4), (10.0 - 5.0 - es.front_dist_ - es.rear_dist_)/20.0, 0.0001);
    EXPECT_NEAR(observed_nn_state(0, 5), (0.0 + 4.0 - es.left_dist_ - es.right_dist_)/10.0, 0.0001);

    // Nearest lon right
    EXPECT_EQ(observed_nn_state(0, 6), 0.0); // longitudinal overlap
    EXPECT_NEAR(observed_nn_state(0, 7), (0.0 -3.0 + es.left_dist_ + es.right_dist_)/10.0, 0.0001);
}

TEST(static_observer, agents_no_left ) { 
    auto params = std::make_shared<SetterParams>();
    params->SetReal("ML::StaticObserver::MinVelLon", -20.0);
    params->SetReal("ML::StaticObserver::MaxVelLon", 20.0);
    params->SetReal("ML::StaticObserver::MinVelLat", -10.0);
    params->SetReal("ML::StaticObserver::MaxVelLat", 10.0);
    params->SetReal("ML::StaticObserver::MaxDist", 100.0);
    params->SetReal("ML::StaticObserver::MinS", -20.0);
    params->SetReal("ML::StaticObserver::MaxS", 20.0);
    params->SetReal("ML::StaticObserver::MinD", -10.0);
    params->SetReal("ML::StaticObserver::MaxD", 10.0);
    params->SetReal("ML::StaticObserver::MinTheta", -B_PI);
    params->SetReal("ML::StaticObserver::MaxTheta", B_PI);
    StaticObserver nn_observer(params);

    const auto world = create_world();
    const auto id1 = create_agent(world, 5.0, left_lane_y, 3.5, 0.0);

    // collides with ego on the right 
    const auto id5 = create_agent(world, 6.0, left_lane_y - 0.1, 3.6, 0.0);

    // behind ego -> should not be considered
    const auto id6 = create_agent(world, 0.0, left_lane_y + 0.0, 3.6, 0.0);

    world->UpdateAgentRTree();
  
    const ObservedWorld observed_world(world, id1);

    const auto observed_nn_state = nn_observer.Observe(observed_world);
    EXPECT_EQ(observed_nn_state.cols(), 8);

    // Ego lat, theta, vlon, vlat
    EXPECT_EQ(observed_nn_state(0, 0), 0.0);
    EXPECT_EQ(observed_nn_state(0, 1), 0.0);
    EXPECT_NEAR(observed_nn_state(0, 2), 3.5/20.0, 0.001);
    EXPECT_EQ(observed_nn_state(0, 3), 0.0);

    const auto es = observed_world.GetEgoAgent()->GetShape();
    // Nearest lon/lat left no existing
    EXPECT_EQ(observed_nn_state(0, 4), -1.0);
    EXPECT_EQ(observed_nn_state(0, 5), -1.0);

    // Nearest lon right
    EXPECT_EQ(observed_nn_state(0, 6), 0.0); // longitudinal overlap
    EXPECT_EQ(observed_nn_state(0, 7), 0.0); // lateral overlap
}
