#include "libtensorflow/include/tensorflow/c/c_api.h"
#include "model_loader.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <string.h>
#include <glog/logging.h>

void NoOpDeallocator(void* data, size_t a, void* b) {}

ModelLoader::ModelLoader(const char* saved_model_dir)
    {
        //********* Read model
        graph_ = TF_NewGraph();
        status_ = TF_NewStatus();
        TF_SessionOptions* session_opts = TF_NewSessionOptions();
        TF_Buffer* run_opts = NULL;
        
        tags = "serve"; 
        
        int ntags = 1; 
        session_ = TF_LoadSessionFromSavedModel(session_opts, run_opts, saved_model_dir, &tags, ntags, graph_, NULL, status_);
        
        if (TF_GetCode(status_) == TF_OK) {
            LOG(INFO) << "TF_LoadSessionFromSavedModel OK\n";
        }
        else {
            LOG(INFO) << "%s" << TF_Message(status_);
        }
    

        //********* Get input tensor
        input_ = (TF_Output*) malloc(sizeof(TF_Output) * num_inputs);
        t0 = {TF_GraphOperationByName(graph_, "serving_default_input"), 0};

        if(t0.oper == NULL) {
            LOG(INFO) << "ERROR: Failed TF_GraphOperationByName serving_default_input\n";
        }
        else {
            LOG(INFO) << "TF_GraphOperationByName serving_default_input is OK\n";
        }
        input_[0] = t0;

        
        //********* Get Output tensor
        output_ = (TF_Output*) malloc(sizeof(TF_Output) * num_outputs);
        t2 = {TF_GraphOperationByName(graph_, "StatefulPartitionedCall"), 0};
        
        if(t2.oper == NULL) {
            LOG(INFO) << "ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n";
        }
        else{
            LOG(INFO) << "TF_GraphOperationByName StatefulPartitionedCall is OK\n";
        }
        output_[0] = t2; 


        //********* Allocate data for inputs & outputs
        input_values  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*num_inputs);
        output_values = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*num_outputs);
        
    }

std::vector<float> ModelLoader::Evaluator(std::vector<float> neural_network_input, int num_actions)
    {        
        int ndims = 2; //ndims is the number of the dimension of neural_network_input
        int len = neural_network_input.size();
        std::vector<std::int64_t> dims = {1, len};
        
        // ndata is total byte size of our data, not the length of the array
        int data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{}); 
        auto data = static_cast<float*>(std::malloc(data_size));
        std::copy(neural_network_input.begin(), neural_network_input.end(), data);
        TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), ndims, data, data_size, &NoOpDeallocator, 0);
        
        if (input_tensor != NULL) {
            LOG(INFO) << "TF_NewTensor is OK\n";
        }
        else{
            LOG(INFO) << "ERROR: Failed TF_NewTensor\n" << std::endl;
        }
        input_values[0] = input_tensor;

        // Run the Session
        TF_SessionRun(session_, NULL, input_, input_values, num_inputs, output_, output_values, num_outputs, NULL, 0, NULL , status_);
        
        if (TF_GetCode(status_) == TF_OK) {
            LOG(INFO) << "Session is OK\n = ";
        }
        else {
            LOG(INFO) << "%s" << TF_Message(status_);
        }

        auto values = (float*) (TF_TensorData(output_values[0]));
        std::vector<float> q_values(values, values + num_actions);

        return q_values;
    }