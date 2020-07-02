from gym.spaces.box import Box


class TF2RLWrapper():
    """class for wrapping a BARK runtime for use as an environment by 
    a tf2rl angent and runner.
    """

    def __init__(self, env):
        """initialize the wrapper"""
        self._env = env
        self.action_space = Box(low=env.action_space.low, high=env.action_space.high)
        self.observation_space = Box(low=env.observation_space.low, high=env.observation_space.high)
    
    def step(self, action):
        """same as in BARK"""
        return self._env.step(action)

    
    def reset(self):
        """same as in BARK"""
        return self._env.reset()


    def render(self, mode=None):
        """for now just return the same rendering independent from mode."""
        return self._env.render()


