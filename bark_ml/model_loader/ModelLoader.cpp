#include "libtensorflow/include/tensorflow/c/c_api.h"
#include "ModelLoader.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <string.h>

void NoOpDeallocator(void* data, size_t a, void* b) {}

ModelLoader::ModelLoader(const char* saved_model_dir)
    {
        //********* Read model
        Graph = TF_NewGraph();
        Status = TF_NewStatus();
        TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
        TF_Buffer* RunOpts = NULL;
        
        tags = "serve"; 
        
        int ntags = 1; 
        Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
        
        if (TF_GetCode(Status) == TF_OK) {
            std::cout << "TF_LoadSessionFromSavedModel OK\n" << std::endl;
        }
        else {
            std::cout << "%s" << TF_Message(Status) << std::endl;
        }
    

        //********* Get input tensor
        Input = (TF_Output*) malloc(sizeof(TF_Output) * NumInputs);
        t0 = {TF_GraphOperationByName(Graph, "serving_default_input"), 0};

        if(t0.oper == NULL) {
            std::cout << "ERROR: Failed TF_GraphOperationByName serving_default_input\n" << std::endl;
        }
        else {
            std::cout << "TF_GraphOperationByName serving_default_input is OK\n" << std::endl;
        }
        Input[0] = t0;

        
        //********* Get Output tensor
        Output = (TF_Output*) malloc(sizeof(TF_Output) * NumOutputs);
        t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
        
        if(t2.oper == NULL) {
            std::cout << "ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n" << std::endl;
        }
        else{
            std::cout << "TF_GraphOperationByName StatefulPartitionedCall is OK\n" << std::endl;
        }
        Output[0] = t2; 


        //********* Allocate data for inputs & outputs
        InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
        OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    }

std::vector<float> ModelLoader::Evaluator(std::vector<float> neural_network_input, int num_actions)
    {        
        int ndims = 2; 
        int len = neural_network_input.size();
        std::vector<std::int64_t> dims = {1, len};
        
        // ndata is total byte size of our data, not the length of the array
        int data_size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<std::int64_t>{}); 
        auto data = static_cast<float*>(std::malloc(data_size));
        std::copy(neural_network_input.begin(), neural_network_input.end(), data);
        TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), ndims, data, data_size, &NoOpDeallocator, 0);
        
        if (input_tensor != NULL) {
            std::cout << "TF_NewTensor is OK\n" << std::endl;
        }
        else{
            std::cout << "ERROR: Failed TF_NewTensor\n" << std::endl;
        }
        InputValues[0] = input_tensor;

        // Run the Session
        TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL , Status);
        
        if (TF_GetCode(Status) == TF_OK) {
            std::cout << "Session is OK\n" << std::endl;
        }
        else {
            std::cout << "%s" << TF_Message(Status) << std::endl;
        }

        auto values = (float*) (TF_TensorData(OutputValues[0]));
        std::vector<float> q_values(values, values + num_actions);

        return q_values;
    }