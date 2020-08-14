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
    TF_Status* status_;
    TF_Graph* graph_;
    const char* tags;
    TF_Session* session_;
    const int num_inputs = 1;
    TF_Output* input_;
    TF_Output t0;
    TF_Output t2;
    const int num_outputs = 1;
    TF_Output* output_;
    TF_Tensor** input_values;
    TF_Tensor** output_values;
};

#endif // SRC_MODEL_LOADER_MODEL_LOADER_HPP_