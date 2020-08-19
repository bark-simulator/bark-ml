#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <algorithm>

#include "bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_wrapper/model_loader.hpp"


#define OBSERVATION_SPACE 16
#define ACTION_SPACE 50

TEST(model_loader, load_module) {

  //std::vector<float> state(OBSERVATION_SPACE);
  //std::generate(std::begin(state), std::end(state),[](){return ((double) rand() / (RAND_MAX));});

  ModelLoader ml("/home/mansoor/Study/Werkstudent/fortiss/code/bark-ml/checkpoints/best/online_net_script.pt", ACTION_SPACE, OBSERVATION_SPACE);
  bool load_status = ml.LoadModel();
  //auto actions = ml.Inference(state);

  //ASSERT_TRUE(load_status);
  //ASSERT_EQ(actions.size(), ACTION_SPACE)
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}