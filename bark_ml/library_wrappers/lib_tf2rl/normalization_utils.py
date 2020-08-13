def rescale(feature, feature_space):
    """Rescales a feature back to their original range."""
    rescaled_feature = (feature + 1.) / 2.
    rescaled_feature *= (feature_space.high - feature_space.low)
    rescaled_feature += feature_space.low
    return rescaled_feature


def normalize(feature, feature_space):
    """Normalizes a feature to be within the range -1 and 1"""
    norm_feature = feature - feature_space.low
    norm_feature /= (feature_space.high - feature_space.low)
    norm_feature = norm_feature * 2. - 1.
    return norm_feature