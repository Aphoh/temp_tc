import wandb
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
api = wandb.Api()
run_length = 100
def analyze_run(run_name):
    run = api.run(run_name)
    hist = run.history()
    reward_prefix = "validation_inner_reward_"
    num_runs = 5
    first_indices = []
    trial_names = []
    for i in range(num_runs):
        trial_name = reward_prefix + str(i)
        if not (trial_name in hist):
            num_runs = i
            break
        else:
            trial_names.append(trial_name)
            first_indices.append(hist[trial_name].first_valid_index())
    run_logs = [run.scan_history(keys=[trial_name]) for trial_name in trial_names]
    reward_vals = np.zeros([num_runs, run_length])
    idx = 0
    for trial in range(num_runs):
        print(trial)
        log_iter = run_logs[trial]
        for idx, log in enumerate(log_iter):
            if idx < run_length:
                reward_vals[trial][idx] = log[trial_names[trial]]
            else:
                print(idx)
    means = np.mean(reward_vals, axis = 0)
    stes = np.std(reward_vals, axis = 0) / np.sqrt(num_runs)
    return means, stes
# Specify which runs to visualize:
# Each dict describes the title of the graph, label of each run, and which wandb runs to plot
runs = {
    "Adaptation from Deterministic Function to Curtail and Shift Response": {
        "MAML+PPO": "social-game-rl/energy-demand-response-game/2g90nma7",
        "PPO": "social-game-rl/energy-demand-response-game/14x4i8so"
    },
    "Adaptation from Linear and Sin Response to Threshold Response": {
        "MAML+PPO": "social-game-rl/energy-demand-response-game/16vrb6mx",
        "PPO": "social-game-rl/energy-demand-response-game/39c9y88s"
        
    },
    # "MAML+PPO Response to Number of Simulation Training Iterations": {
    #     "50 inner step iterations": "social-game-rl/energy-demand-response-game/24nd2dj1",
    #     "100 inner step iterations": "social-game-rl/energy-demand-response-game/2g90nma7",
    #     "150 inner step iterations": "social-game-rl/energy-demand-response-game/3tc4ni7u",
    #     "200 inner step iterations": "social-game-rl/energy-demand-response-game/2ev8r466"
    # }
}
plt.rcParams.update({'font.size': 32})
plt.rcParams['axes.linewidth'] = 3 # set the value globally
fig, axs = plt.subplots(len(runs.keys()), sharex=True, figsize=(20, 20))
colors = ["g", "r", "c", "m"]
for i, (name, wandb_ids) in enumerate(runs.items()):
    if len(wandb_ids.keys()) > 2:
        ax = axs
    else:
        ax = axs[i]
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 1 decimal place
    for j, (algo, id) in enumerate(wandb_ids.items()):
        means, stes = analyze_run(id)
        x = list(range(len(means)))
        if len(wandb_ids.keys()) > 2:
            ax.errorbar(x, means, yerr = stes, label=algo, linewidth=3.0, color=colors[j])
        else:
            ax.errorbar(x, means, yerr = stes, label=algo, linewidth=3.0)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.set_title(name)
    ax.set_ylabel("Average Reward", fontsize=40)
    if i == 0:
        ax.legend()
plt.xlabel("Gradient Update Steps", fontsize=40)

fig.tight_layout()
plt.savefig("maml_summary_perf_grid.png")