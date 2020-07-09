from gym.spaces.box import Box


class TF2RLWrapper():
    """class for wrapping a BARK runtime for use as an environment by 
    a tf2rl angent and runner.
    """

    def __init__(self, env, normalize_features=True):
        """initialize the wrapper"""
        self._env = env
        self._normalize_features = normalize_features

        if self._normalize_features:
            self.action_space = Box(low=-1, high=1, shape=env.action_space.shape)
            self.observation_space = Box(low=-1, high=1, shape=env.observation_space.shape)
        else:
            self.action_space = Box(low=env.action_space.low, high=env.action_space.high)
            self.observation_space = Box(low=env.observation_space.low, high=env.observation_space.high)

    
    def step(self, action):
        """step function of the environment. 
        - If normalization is needed: The action is scaled back to its original range, fed to the runtime
            and then the observation is normalized before returning.
        - If normalization is not needed: Simply calls the step function of the runtime.
        """
        if self._normalize_features:
            rescaled_action = self._rescale_action(action.copy())
            next_obs, reward, done, _ = self._env.step(rescaled_action)
            next_obs = self._normalize_observation(next_obs.copy())
            return next_obs, reward, done, None
        else
            return self._env.step(action)

    
    def reset(self):
        """reset function of the environment. 
        - If normalization is needed: The observation is normalized before returning.
        - If normalization is not needed: Simply calls the reset function of the runtime.
        """
        if self._normalize_features:
            return self._normalize_observation(self._env.reset().copy())
        else:
            return self._env.reset()


    def render(self, mode=None):
        """for now just return the same rendering independent from mode."""
        return self._env.render()


    def _rescale_action(self, action):
        """rescales a normalized action back to their original range."""
        action = (action + 1.) / 2.
        action *= (self._env.action_space.high - self._env.action_space.low)
        action += self._env.action_space.low
        return action


    def _normalize_observation(self, observation):
        """Normalizes an observation to be within the range -1 and 1"""
        observation -= self._env.observation_space.low
        observation /= (self._env.observation_space.high - self._env.observation_space.low)
        observation = observation * 2. - 1.
        return observation

