// Copyright (c) fortiss GmbH
//
// Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_EVALUATORS_GOAL_REACHED_HPP_
#define BARK_ML_EVALUATORS_GOAL_REACHED_HPP_

#include <memory>
#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <functional>
#include <Eigen/Dense>
#include <boost/variant.hpp>

#include "bark_ml/evaluators/base_evaluator.hpp"
#include "bark/world/evaluation/evaluator_goal_reached.hpp"
#include "bark/world/evaluation/evaluator_step_count.hpp"
#include "bark/world/evaluation/evaluator_collision_ego_agent.hpp"
#include "bark/world/evaluation/evaluator_drivable_area.hpp"

namespace bark_ml {
namespace evaluators {
using bark::commons::ParamsPtr;
using bark::world::Agent;
using spaces::Box;
using commons::Norm;
using spaces::Matrix_t;
using bark::world::AgentMap;
using bark::world::AgentPtr;
using bark::world::WorldPtr;
using bark::world::evaluation::EvaluationReturn;
using bark::world::evaluation::EvaluatorGoalReached;
using bark::world::evaluation::EvaluatorCollisionEgoAgent;
using bark::world::evaluation::EvaluatorStepCount;
using bark::world::evaluation::EvaluatorDrivableArea;
using bark::world::evaluation::EvaluatorPtr;
using bark::world::goal_definition::GoalDefinitionStateLimitsFrenet;
using bark::world::ObservedWorldPtr;
using bark::geometry::Point2d;
using bark::geometry::Line;
using bark::geometry::Distance;
using bark::geometry::Norm0To2PI;
using bark::models::dynamic::Input;
using bark::models::dynamic::StateDefinition::X_POSITION;
using bark::models::dynamic::StateDefinition::Y_POSITION;
using bark::models::dynamic::StateDefinition::THETA_POSITION;
using bark::models::dynamic::StateDefinition::VEL_POSITION;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using bark::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using bark_ml::evaluators::BaseEvaluator;

using Reward = double;
using Done = bool;
using EvalResults = std::map<std::string, EvaluationReturn>;

/**
 * @brief Sparse reward evaluator returning +1 for reaching the goal,
  -1 for having a collision or leaving the drivable area.
 */
class GoalReachedEvaluator : public BaseEvaluator {
 public:
  explicit GoalReachedEvaluator(const ParamsPtr& params) :
    BaseEvaluator(params) {
    col_penalty_ = params->GetReal(
      "ML::GoalReachedEvaluator::ColPenalty", "", -1.);
    goal_reward_ = params->GetReal(
      "ML::GoalReachedEvaluator::GoalReward", "", 1.);
    max_steps_ = params->GetInt(
      "ML::GoalReachedEvaluator::MaxSteps", "", 50);
  }

  void AddEvaluators(WorldPtr& world) {  // NOLINT
    world->AddEvaluator("goal_reached",
      std::make_shared<EvaluatorGoalReached>());
    world->AddEvaluator("collision",
      std::make_shared<EvaluatorCollisionEgoAgent>());
    world->AddEvaluator("step_count",
      std::make_shared<EvaluatorStepCount>());
    world->AddEvaluator("drivable_area",
      std::make_shared<EvaluatorDrivableArea>());
  }

  std::tuple<Reward, Done, EvalResults> Evaluate(
    const ObservedWorldPtr& observed_world,
    const Input& input) const {
      EvalResults eval_results = observed_world->Evaluate();
      bool is_terminal = false;
      bool success = boost::get<bool>(eval_results["goal_reached"]);
      bool collision = boost::get<bool>(eval_results["collision"]) ||
                       boost::get<bool>(eval_results["drivable_area"]);
      int step_count = boost::get<int>(eval_results["step_count"]);
      if (success || collision || step_count > max_steps_)
        is_terminal = true;
      if (collision)
        success = 0;
      double reward = collision * col_penalty_ + success * goal_reward_;
      return {reward, is_terminal, eval_results};
    }

  WorldPtr Reset(WorldPtr& world) {  // NOLINT
    world->ClearEvaluators();
    AddEvaluators(world);
    return world;
  }

 private:
  double col_penalty_, goal_reward_;
  int max_steps_;
};

}  // namespace evaluators
}  // namespace bark_ml

#endif  // BARK_ML_EVALUATORS_GOAL_REACHED_HPP_
