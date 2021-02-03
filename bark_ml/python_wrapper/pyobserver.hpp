// Copyright (c) 2020 fortiss GmbH
//
// Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
// Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#ifndef BARK_ML_PYTHON_WRAPPER_PYOBSERVER_HPP_
#define BARK_ML_PYTHON_WRAPPER_PYOBSERVER_HPP_
#include <memory>

#include "bark_ml/observers/base_observer.hpp"

namespace py = pybind11;
using bark_ml::observers::BaseObserver;
using bark_ml::spaces::Box;
using bark::world::WorldPtr;
using bark::world::ObservedWorldPtr;
using ObservedState = Eigen::Matrix<float, 1, Eigen::Dynamic>;

class PyObserver : public BaseObserver {
 public:
  using BaseObserver::BaseObserver;

  ObservedState Observe(
    const ObservedWorldPtr& observed_world) const {
    PYBIND11_OVERLOAD_PURE(ObservedState, BaseObserver, Observe,
                           observed_world); }

  WorldPtr Reset(const WorldPtr& world) {
    PYBIND11_OVERLOAD_PURE(WorldPtr, BaseObserver, Reset,
                           world); }
  Box<float> ObservationSpace() const {
    PYBIND11_OVERLOAD_PURE(Box<float>, BaseObserver, ObservationSpace,
                           ); }

};


#endif  // BARK_ML_PYTHON_WRAPPER_PYOBSERVER_HPP_
