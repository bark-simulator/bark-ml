import numpy as np
import matplotlib.pyplot as plt
import pandas

from bark_ml.commons.tracer import Tracer

runner_tracer = Tracer()
runner_tracer.Load("/Users/hart/Development/bark-ml/results/data/evaluation_results_runner.pckl")

runtime_tracer = Tracer()
runtime_tracer.Load("/Users/hart/Development/bark-ml/results/data/evaluation_results_runtime.pckl")


goal_reached = runner_tracer.success_rate
col_rate = runner_tracer.collision_rate
exc_pol = runtime_tracer.Query(
  key="executed_learned_policy", group_by="max_col_rate", agg_type="MEAN")

max_col_rates = [0., 0.25, 0.5, 0.75, 1.]

goal_reached = [goal_reached.iloc[goal_reached.index.get_level_values("max_col_rate") == cr].mean() for cr in max_col_rates]
col_rate = [col_rate.iloc[col_rate.index.get_level_values("max_col_rate") == cr].mean() for cr in max_col_rates]
plt.plot(max_col_rates, col_rate, label="Collision-rate", color="black", linestyle="dashed")
plt.plot(max_col_rates, goal_reached, label="Success-rate", color="black", linestyle="solid")
plt.plot(max_col_rates, exc_pol, label="Execution-rate of $\pi_{ego}$", color="black", linestyle="dotted")
plt.xlabel("Max. allowed risk [%]")
plt.ylabel("%")
plt.legend()
plt.show()
# NOTE: plot stuff