import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.pyplot import figure


log_dir = "logs/"
log_names = os.listdir(log_dir)
figure(figsize=(20, 20))
prefixes = {
    "Pretrained SAC 100 Steps": "pretrained_sac_nosmirl_nodagger_ablation_sin_1_",
    "Pretrained SAC 256x3 Steps": "pretrained_sac_nosmirl_nodagger_ablation_1_",
    #"Pretrained SAC 512x3 Steps": "pretrained_sac_nosmirl_nodagger_ablation_2_",
    #"Pretrained SAC 768x3 Steps": "pretrained_sac_nosmirl_nodagger_ablation_3_",
    "Pretrained SAC 44800x3 Steps": "pretrained_sac_nosmirl_nodagger_ablation_175_",
    #"Pretrained (SMiRL) SAC (w/ planning)": "pretrained_sac_smirl_1000dagger_redo",
    #"Pretrained (SMiRL) SAC (no planning)": "pretrained_sac_smirl_nodagger_redo",
    #"Vanilla SAC (w/ planning) (2 Stage Dagger)": "vanilla_sac_1000dagger_2stage",
    #"Vanilla SAC (w/ planning)": "vanilla_sac_1000dagger",
    #"Vanilla SAC (no planning)": "vanilla_sac_nodagger",
    #"SMiRL SAC (w/ planning) (2 Stage Dagger)": "smirl_sac_1000dagger_2stage",
    #"SMiRL SAC (w/ planning)": "smirl_sac_1000dagger",
    #"SMiRL SAC (no planning)": "smirl_sac_nodagger",
}
graphs = { exp_name: 
    [os.path.join(log_dir, log_name, "progress.csv") for log_name in log_names if log_name[:len(prefix)] == prefix and log_name[len(prefix)].isdigit() ] 
    for exp_name, prefix in prefixes.items()
}

# graphs = {
#     "Pretrained SAC (no planning)": "logs/pretrained_sac_nodagger_2021-06-22 02:34:09.373771/progress.csv",
#     "Pretrained SAC (planning)": "logs/pretrained_sac_dagger_2021-06-16 22:58:27.262384/progress.csv",
#     "Vanilla SAC (planning)": "logs/vanilla_sac_dagger_2021-06-22 00:15:51.410218/progress.csv",
#     "Vanilla SAC (no planning)": "logs/vanilla_sac_nodagger_2021-06-24 04:57:19.839638/progress.csv",
#     "Pretrained (SMiRL) SAC (planning)": "logs/pretrainedsmirl_sac_dagger_2021-06-24 06:38:59.830893/progress.csv",
#     "Pretrained (SMiRL) SAC (no planning)": "logs/pretrainedsmirl_sac_nodagger_2021-06-24 07:42:36.141628/progress.csv"
# }
x_cap = 4000
min_diff = 100
plt.ylabel("Average Daily Cost")
plt.xlabel("Number of Steps Sampled")
plt.title("Analysis of Pretraining, SMiRL, and Dagger")
def plot_graph(ax, label, paths):
    energy_costs = [[] for _ in range(5)]

    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        energy_cost = df["custom_metrics/energy_cost_mean"]
        if "custom_metrics/real_step_max" not in df.keys() or df["custom_metrics/real_step_max"].isnull().values.any():
            steps = df["info/num_steps_sampled"]
        else:
            steps = df["custom_metrics/real_step_max"]

        idx = steps < x_cap
        energy_cost = energy_cost[idx]
        steps = steps[idx]
        idx = [True]
        last_step = steps[0]
        for step in steps[1:]:
            if (step - last_step) > min_diff:
                idx.append(True)
                last_step = step
            else:
                idx.append(False)
        #idx = [True] + ((steps[1:] - steps[:-1]) > min_diff)
        steps = steps[idx]
        energy_cost = energy_cost[idx]
        print(steps.max(), len(steps))
        energy_costs[i] = (energy_cost.to_numpy() + 40) * 250
    
    energy_costs = np.array(energy_costs)
    if "w/ planning" in label and "2 Stage Dagger" not in label:
        steps += 1000 # shift by 1000 steps to account for training time
    ax.errorbar(steps, energy_costs.mean(axis=0), energy_costs.std(axis=0)/np.sqrt(5), label=label)
    return steps
    #ax.plot(steps, energy_costs, label=label)

def plot_baselines(ax, steps):
    ax.plot(steps, np.array([(75 + 40) * 250 for step in steps]), label="TOU")
    ax.plot(steps, np.array([(120) * 250 for step in steps]), label="Baseline")

for label, paths in graphs.items():
    steps = plot_graph(plt, label, paths)

plot_baselines(plt, list(range(x_cap)))
plt.legend()
plt.savefig("2stage.png")