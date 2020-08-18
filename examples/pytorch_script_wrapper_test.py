#include "examples/pytorch_script_wrapper.hpp"
import bark_ml.examples.pytorch_script_wrapper as test
import numpy as np

actions = test.predict("/home/mansoor/Study/Werkstudent/fortiss/code/bark-ml/checkpoints/best/online_net_script.pt", np.random.rand(16).tolist())
print (actions)