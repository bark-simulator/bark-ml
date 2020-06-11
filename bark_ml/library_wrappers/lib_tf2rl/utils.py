import os
import pickle

def load_expert_trajectory_file(filepath: str):
    """
    Loads the given expert trajectory pickle file.
    """
    if not os.path.exists(filepath):
        raise ValueError(f"{filepath} does not exist.")

    if not filepath.endswith(".pkl"):
        raise ValueError(f"{filepath} is not a .pkl file.")

    with open(filepath, "rb") as pickle_file:
        expert_trajectories = pickle.load(pickle_file)
        return expert_trajectories