// #include <torch/script.h> // One-stop header.
// #include <iostream>
// #include <memory>

// #include "bark/world/observed_world.hpp"
// #include "bark/commons/params/setter_params.hpp"
// #include "bark/commons/params/params.hpp"
// #include "bark/geometry/polygon.hpp"
// #include "bark/geometry/standard_shapes.hpp"
// #include "bark/models/behavior/constant_velocity/constant_velocity.hpp"
// #include "bark/models/behavior/motion_primitives/continuous_actions.hpp"
// #include "bark/models/dynamic/single_track.hpp"
// #include "bark/models/execution/interpolation/interpolate.hpp"
// #include "bark/world/evaluation/evaluator_collision_agents.hpp"
// #include "bark/world/goal_definition/goal_definition.hpp"
// #include "bark/world/goal_definition/goal_definition_polygon.hpp"
// #include "bark/world/map/map_interface.hpp"
// #include "bark/world/map/roadgraph.hpp"
// #include "bark/world/objects/agent.hpp"
// #include "bark/world/opendrive/opendrive.hpp"
// #include "bark/world/tests/dummy_road_corridor.hpp"
// #include "bark/world/tests/make_test_world.hpp"
// #include "bark/world/tests/make_test_xodr_map.hpp"

// #include "bark_ml/observers/nearest_observer.hpp"

// using namespace bark::models::dynamic;
// using namespace bark::models::behavior;
// using namespace bark::models::execution;
// using namespace bark::world::map;

// using bark::commons::SetterParams;
// using bark::commons::Params;
// using bark::commons::transformation::FrenetPosition;
// using bark::geometry::Model3D;
// using bark::geometry::Point2d;
// using bark::geometry::Polygon;
// using bark::geometry::Pose;
// using bark::geometry::standard_shapes::CarRectangle;
// using bark::world::FrontRearAgents;
// using bark::world::ObservedWorld;
// using bark::world::ObservedWorldPtr;
// using bark::world::World;
// using bark::world::WorldPtr;
// using bark::world::goal_definition::GoalDefinitionPolygon;
// using bark::world::objects::Agent;
// using bark::world::objects::AgentPtr;
// using bark::world::opendrive::OpenDriveMapPtr;
// using bark::world::tests::MakeXodrMapOneRoadTwoLanes;
// using StateDefinition::MIN_STATE_SIZE;
// using bark_ml::observers::NearestObserver;

// class BehaviorDiscreteML : public BehaviorMPContinuousActions {

//  public:
//   BehaviorDiscreteML(const ParamsPtr& params)
//       : BehaviorMPContinuousActions(params){
//         auto acc_min_max = params->GetListFloat("ML::BehaviorDiscreteML::MinMaxAcc", "", {-3.0, 3.0});
//         auto steer_min_max = params->GetListFloat("ML::BehaviorDiscreteML::MinMaxSteeringRate", "", {-2.0, 2.0});

//         auto accelerations = linespace(acc_min_max[0], acc_min_max[1], 10);
//         auto steerings = linespace(steer_min_max[0], steer_min_max[1], 5);

//         for(const float &acc: accelerations) {
//           for (const float &steer: steerings) {
//             Input i(2);
//             i << acc, steer;
//             AddMotionPrimitive(i);
//           }
//         }
//       }

//   virtual ~BehaviorDiscreteML() {}

//   private:
//     template <class T>
//     std::vector<T> linespace(T start, T ed, int num) {
//         // catch rarely, throw often
//         int partitions = num - 1;
//         std::vector<T> pts;
//         // length of each segment    
//         T length = (ed - start) / partitions; 
//         // first, not to change
//         pts.push_back(start);
//         for (int i = 1; i < num - 1; i ++) {
//             pts.push_back(start + i * length);
//         }
//         // last, not to change
//         pts.push_back(ed);
//         return pts;
//     }
// };

// // Load a pytorch trained network (nn.Module) saved from
// // python and perform an inference

// int main(int argc, const char* argv[]) {
//   if (argc != 2) {
//     std::cerr << "usage: bazel run //examples:pytorch_model_cpp_observer <model_path>\n";
//     return -1;
//   }

//   torch::jit::script::Module module;
//   try {
//     // deserialize the ScriptModule from a saved python trained model using torch::jit::load().
//     module = torch::jit::load(argv[1]);
//   }
//   catch (const c10::Error& e) {
//     std::cerr << "Error loading the model\n";
//     return -1;
//   }

//   auto params = std::make_shared<SetterParams>();

//   // Setting Up Map
//   OpenDriveMapPtr open_drive_map = MakeXodrMapOneRoadTwoLanes();
//   MapInterfacePtr map_interface = std::make_shared<MapInterface>();
//   map_interface->interface_from_opendrive(open_drive_map);

//   // Goal Definition
//   Polygon polygon(Pose(1, 1, 0),
//       std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(2, 2),Point2d(2, 0), Point2d(0, 0)});
//   std::shared_ptr<Polygon> goal_polygon(
//       std::dynamic_pointer_cast<Polygon>(polygon.Translate(Point2d(50, -2))));
//   auto goal_ptr = std::make_shared<GoalDefinitionPolygon>(*goal_polygon);

//   // Setting Up Agents (one in front of another)
//   ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
//   DynamicModelPtr dyn_model(new SingleTrackModel(params));
//   BehaviorModelPtr beh_model(new BehaviorConstantVelocity(params));
//   Polygon car_polygon = CarRectangle();

//   State init_state1(static_cast<int>(MIN_STATE_SIZE));
//   init_state1 << 0.0, 3.0, -1.75, 0.0, 5.0;
//   AgentPtr agent1(new Agent(init_state1, beh_model, dyn_model, exec_model,
//                             car_polygon, params, goal_ptr, map_interface,
//                             Model3D()));  // NOLINT

//   State init_state2(static_cast<int>(MIN_STATE_SIZE));
//   init_state2 << 0.0, 10.0, -1.75, 0.0, 5.0;
//   // discrete behavior model for ML
//   std::shared_ptr<BehaviorDiscreteML> discrete_beh_model(new BehaviorDiscreteML(params));

//   AgentPtr agent2(new Agent(init_state2, discrete_beh_model, dyn_model, exec_model,
//                             car_polygon, params, goal_ptr, map_interface, Model3D()));  

//   // construct World
//   WorldPtr world(new World(params));
//   world->AddAgent(agent1);
//   world->AddAgent(agent2);
//   world->UpdateAgentRTree();

//   // setup observed world 
//   WorldPtr current_world_state1(world->Clone());
//   ObservedWorldPtr obs_world1(new ObservedWorld(current_world_state1, agent2->GetAgentId()));

//   std::shared_ptr<NearestObserver> observer(new NearestObserver(params));

//   // setup nearest observer
//   observer->Reset(world);
//   auto os = observer->ObservationSpace();
//   //std::cout<<"observation_space:"<<std::get<int>(os.shape())<<std::endl;
//   auto observedStateMatrix = observer->Observe(obs_world1);
//   std::vector<float> observedState(observedStateMatrix.data(), observedStateMatrix.data()+observedStateMatrix.size());

//   // pass the current state to model for  action inference
//   std::vector<torch::jit::IValue> inputs;
//   inputs.push_back(torch::tensor(observedState));
  
//   at::Tensor actions;
//   try {
//     // run the inference
//     actions = module.forward(inputs).toTensor();

//   } catch (const c10::Error& e) {
//     std::cerr << e.msg();
//     return -1;
//   }

//   // pick the action with maximum reward
//   auto maxRewardAction = *torch::argmax(actions).data<int64_t>();
//   std::cout << "Action:"<<maxRewardAction<<std::endl;

//   // apply action to agent
//   BehaviorMPContinuousActions::MotionIdx idx = maxRewardAction;
//   world->GetAgent(agent2->GetAgentId())->GetBehaviorModel()->ActionToBehavior(idx);

//   // step to next state
//   const float dt = 0.01;
//   world->Step(dt);
//   WorldPtr current_world_state2(world->Clone());
//   ObservedWorldPtr obs_world2(new ObservedWorld(current_world_state2, agent2->GetAgentId()));
  
//   // next state after taking the predicted action
//   auto nextObservedState = observer->Observe(obs_world1);
  
//   //loop - todo
//   return 0;
// }
