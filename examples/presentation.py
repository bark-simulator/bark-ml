from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories


expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories("/home/brucknem/Repositories/gail-4-bark/bark-ml/examples/expert_trajectories/DR_DEU_Merging_MT_v01_shifted") 

print(avg_trajectory_length)
print(num_trajectories)
