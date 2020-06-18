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
        expert_trajectories = dict(pickle.load(pickle_file))

    for key, value in expert_trajectories.items():
        if ("obs" not in value or
                "act" not in value or
                "done" not in value or
                "time" not in value or
                "merge" not in value):
            raise ValueError(f"{filepath} does not contain valid "
                             "expert trajectories.\n"
                             f"{key}, {value} are invalid.")

    return expert_trajectories
