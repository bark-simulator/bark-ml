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
  }

  virtual void AddEvaluators(WorldPtr& world) {
    
  }

  virtual std::tuple<Reward, Done, EvalResults> Evaluate(
    const ObservedWorldPtr& observed_world,
    const Input& input) const {
      EvalResults eval_res;
      eval_res["eval"] = 1;
      return {Reward(0), Done(0), eval_res};
    }

  WorldPtr Reset(WorldPtr& world) {
    world->ClearEvaluators();
    AddEvaluators(world);
    return world;
  }
};

}  // namespace evaluators

#endif  // BARK_ML_EVALUATORS_GOAL_REACHED_HPP_
