#include "libtensorflow/include/tensorflow/c/c_api.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <random>
#include "bark_ml/model_loader/ModelLoader.hpp"


int main()
{
    ModelLoader model("/home/model/");

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    std::vector<float> input(16);
    for (int i=0;i<16;i++)
    {
        input[i] = distribution(generator);
    };
    
    std::vector<float> q_values = model.Evaluator(input, 8);

    for (int i=0;i<8;i++)
    {
        std::cout << q_values[i] << std::endl;
    }
    
    return 0;

}