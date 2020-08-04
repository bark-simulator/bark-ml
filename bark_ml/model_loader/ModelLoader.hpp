// Copyright (c) 2020 fortiss GmbH
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#ifndef SRC_MODEL_LOADER_MODEL_LOADER_HPP_
#define SRC_MODEL_LOADER_MODEL_LOADER_HPP_

#include "libtensorflow/include/tensorflow/c/c_api.h"
#include <vector>


inline void NoOpDeallocator() {};

class ModelLoader {
public:
    ModelLoader(const char* saved_model_dir);
    std::vector<float> Evaluator(std::vector<float> neural_network_input, int num_actions);

private:
    TF_Status* Status;
    TF_Graph* Graph;
    const char* tags;
    TF_Session* Session;
    const int NumInputs = 1;
    TF_Output* Input;
    TF_Output t0;
    TF_Output t2;
    const int NumOutputs = 1;
    TF_Output* Output;
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
};

#endif // SRC_MODEL_LOADER_MODEL_LOADER_HPP_