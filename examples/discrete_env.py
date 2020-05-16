import gym
import numpy as np
import bark_ml.environments.gym

# cont. highway env
env = gym.make("highway-v1")
env.reset()

done = False
while done is False:
  action = np.random.randint(low=0, high=3)
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")