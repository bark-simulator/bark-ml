"""In case a wrapper is needed around the bark runtime for be usable by tf2rl
as a simple environment. So far the only thing that was found is that if 
args.save_test_movie is True, that is a gif is saved about the testing, the 
render() function is called with an extra argument mode=\"rgb_array\". 
The bark runtime does not except such an argument, so if the movie saving
function is needed to be used then a wrapper around this is needed.
"""


class TF2RLWrapper():
    """class for wrapping a BARK runtime for use as an environment by 
    a tf2rl angent and runner.
    """

    def __init__(self, env):

        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def step(self, action):
        return self._env.step(action)

    
    def reset(self):
        return self._env.reset()


    def render(self, mode=None):
        """for now just return the same rendering independent from mode."""
        return self._env.render()


