// Copyright (c) Patrick Hart, Julian Bernhard,
// Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#ifndef BARK_ML_EVALUATORS_BASE_EVALUATOR_HPP_
#define BARK_ML_EVALUATORS_BASE_EVALUATOR_HPP_

#include <memory>
#include <vector>
#include <tuple>
#include <map>
#include <functional>
#include <Eigen/Dense>
#include <boost/variant.hpp>

#include "bark/commons/params/params.hpp"
#include "bark/world/world.hpp"
#include "bark/world/observed_world.hpp"
#include "bark/world/goal_definition/goal_definition_state_limits_frenet.hpp"
#include "bark/models/dynamic/dynamic_model.hpp"
#include "bark_ml/commons/spaces.hpp"
#include "bark_ml/commons/commons.hpp"

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
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using Reward = float;
using Done = bool;
using EvalResults = std::map<std::string, EvaluationReturn>;


class BaseEvaluator {
 public:
  explicit BaseEvaluator(const ParamsPtr& params) :
    params_(params) {
  }

  virtual void AddEvaluators(WorldPtr& world) = 0;

  virtual std::tuple<Reward, Done, EvalResults> Evaluate(
    const ObservedWorldPtr& observed_world,
    const Input& input) const = 0;

  WorldPtr Reset(WorldPtr& world) {
    world->ClearEvaluators();
    AddEvaluators(world);
    return world;
  }

 private:
  ParamsPtr params_;
};

}  // namespace evaluators
}  // namespace bark_ml

#endif  // BARK_ML_EVALUATORS_BASE_EVALUATOR_HPP_
