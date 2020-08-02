#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

// Load a pytorch trained network (nn.Module) saved from
// python and perform an inference

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: bazel run //examples:pytorch_model_cpp <model_path>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // deserialize the ScriptModule from a saved python trained model using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return -1;
  }

  // wrap environment observation for inference
  const u_int64_t OBSERVATION_SPACE = 16;
  const u_int64_t ACTION_SPACE = 50;
  const float MIN_REWARD = -1;
  const float MAX_REWARD = 1;

  std::vector<torch::jit::IValue> inputs;

  // sample a random sample from observation_space
  auto observation = torch::rand({1, OBSERVATION_SPACE});
  inputs.push_back(observation);
  
  at::Tensor actions;
  try {
    // run the inference
    actions = module.forward(inputs).toTensor();

  } catch (const c10::Error& e) {
    std::cerr << e.msg();
    return -1;
  }

  // verify the output range and shape
  assert(actions.sizes()[1] == ACTION_SPACE);
  assert(actions.min().item().toFloat() >= MIN_REWARD);
  assert(actions.max().item().toFloat() <= MAX_REWARD);

  // pick the action with maximum reward
  auto maxRewardAction = *torch::argmax(actions).data<int64_t>();
  std::cout << "Action:"<<maxRewardAction<<std::endl;

  return 0;
}
