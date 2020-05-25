// Copyright (c) Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
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
#include "modules/world/evaluation/evaluator_goal_reached.hpp"
#include "modules/world/evaluation/evaluator_step_count.hpp"
#include "modules/world/evaluation/evaluator_collision_ego_agent.hpp"
#include "modules/world/evaluation/evaluator_drivable_area.hpp"

namespace evaluators {
using modules::commons::ParamsPtr;
using modules::world::Agent;
using spaces::Box;
using commons::Norm;
using spaces::Matrix_t;
using modules::world::AgentMap;
using modules::world::AgentPtr;
using modules::world::WorldPtr;
using modules::world::evaluation::EvaluationReturn;
using modules::world::evaluation::EvaluatorGoalReached;
using modules::world::evaluation::EvaluatorCollisionEgoAgent;
using modules::world::evaluation::EvaluatorStepCount;
using modules::world::evaluation::EvaluatorDrivableArea;
using modules::world::evaluation::EvaluatorPtr;
using modules::world::goal_definition::GoalDefinitionStateLimitsFrenet;
using modules::world::ObservedWorldPtr;
using modules::geometry::Point2d;
using modules::geometry::Line;
using modules::geometry::Distance;
using modules::geometry::Norm0To2PI;
using modules::models::dynamic::Input;
using modules::models::dynamic::StateDefinition::X_POSITION;
using modules::models::dynamic::StateDefinition::Y_POSITION;
using modules::models::dynamic::StateDefinition::THETA_POSITION;
using modules::models::dynamic::StateDefinition::VEL_POSITION;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using modules::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using Reward = float;
using Done = bool;
using EvalResults = std::map<std::string, EvaluationReturn>;


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
      float reward = collision * col_penalty_ + success * goal_reward_;
      return {reward, is_terminal, eval_results};
    }

  WorldPtr Reset(WorldPtr& world) {  // NOLINT
    world->ClearEvaluators();
    AddEvaluators(world);
    return world;
  }

 private:
  float col_penalty_, goal_reward_;
  int max_steps_;
};

}  // namespace evaluators

#endif  // BARK_ML_EVALUATORS_GOAL_REACHED_HPP_
