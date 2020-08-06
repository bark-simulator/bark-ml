def rescale_action(action, action_space):
    """rescales a normalized action back to their original range."""
    rescaled_action = (action + 1.) / 2.
    rescaled_action *= (action_space.high - action_space.low)
    rescaled_action += action_space.low
    return rescaled_action


def normalize_observation(observation, observation_space):
    """Normalizes an observation to be within the range -1 and 1"""
    norm_observation = observation - observation_space.low
    norm_observation /= (observation_space.high - observation_space.low)
    norm_observation = norm_observation * 2. - 1.
    return norm_observation