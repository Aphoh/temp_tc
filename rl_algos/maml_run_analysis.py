import wandb
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
api = wandb.Api()
run_length = 1000
def analyze_run(run_name, run_length):
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

def analyze_offline_runs(run_name):
    run = api.run(run_name)
    hist = run.history()
    reward_vals = np.ones([run_length]) * -4.43
    key = "environment_reward"
    run_log = run.scan_history(keys=[key])
    log_iter = run_log
    for idx, log in enumerate(log_iter):
        if idx < run_length:
            if log[key] != 0:
                reward_vals[idx] = log[key]
            else:
                reward_vals[idx] = -4.43
        else:
            print(idx)
            break
    return pd.DataFrame(reward_vals).ewm(com=10).mean(), None
# Specify which runs to visualize:
# Each dict describes the title of the graph, label of each run, and which wandb runs to plot
runs = {
    # "Adaptation from Deterministic Function to Curtail and Shift Response": {
    #     "MAML+PPO": "social-game-rl/energy-demand-response-game/2g90nma7",
    #     "PPO": "social-game-rl/energy-demand-response-game/14x4i8so"
    # },
    # "Adaptation from Linear and Sin Response to Threshold Response": {
    #     "MAML+PPO": "social-game-rl/energy-demand-response-game/16vrb6mx",
    #     "PPO": "social-game-rl/energy-demand-response-game/39c9y88s"
        
    # },
    # "MAML+PPO Response to Number of Simulation Training Iterations": {
    #     "50 inner step iterations": "social-game-rl/energy-demand-response-game/24nd2dj1",
    #     "100 inner step iterations": "social-game-rl/energy-demand-response-game/2g90nma7",
    #     "150 inner step iterations": "social-game-rl/energy-demand-response-game/3tc4ni7u",
    #     "200 inner step iterations": "social-game-rl/energy-demand-response-game/2ev8r466"
    # }
    "SAC Response to Perturbed Offline Dataset Mix-In": {
        "0.9 (Unperturbed)": "social-game-rl/energy-demand-response-game/1biixllh",
        "0.7 (Unperturbed)": "social-game-rl/energy-demand-response-game/22871kay",
        "0.5 (Unperturbed)": "social-game-rl/energy-demand-response-game/35f9qwb0",
        
        "0.9 (Perturbed)": "social-game-rl/energy-demand-response-game/2xt3fk7o",
        "0.7 (Perturbed)": "social-game-rl/energy-demand-response-game/3w2w1hc7",
        "0.5 (Perturbed)": "social-game-rl/energy-demand-response-game/2u49jnb8"

    },
    "Pretrained SAC vs Mix-In SAC": {
        "Pretrained": "social-game-rl/energy-demand-response-game/28jiatu6",
        "0.99": "social-game-rl/energy-demand-response-game/3o3oz1e7",
        "0.95": "social-game-rl/energy-demand-response-game/q2yq7r4z",
        "0.9": "social-game-rl/energy-demand-response-game/2xt3fk7o",
        "0.7": "social-game-rl/energy-demand-response-game/3w2w1hc7",
        "0.5": "social-game-rl/energy-demand-response-game/2u49jnb8",
    },
    "MAML+PPO vs Pretrained SAC": {
        "PPO": "social-game-rl/energy-demand-response-game/14x4i8so",
        "MAML+PPO": "social-game-rl/energy-demand-response-game/2g90nma7",
        "SAC (Vanilla)": "social-game-rl/energy-demand-response-game/1n155f7q",
        "SAC (Pretrained)": "social-game-rl/energy-demand-response-game/3eo7en6e",
        
        #"SAC (0.9 Offline)": "social-game-rl/energy-demand-response-game/2xt3fk7o",
        #"SAC (0.7 Offline)": "social-game-rl/energy-demand-response-game/3w2w1hc7"
    }
}
run_means = {key: {} for key, val in runs.items()}
plt.rcParams.update({'font.size': 32})
plt.rcParams['axes.linewidth'] = 3 # set the value globally
fig, axs = plt.subplots(len(runs.keys()), sharex=True, figsize=(20, 20))
colors = ["g", "r", "c", "m"]
for i, (name, wandb_ids) in enumerate(runs.items()):
    # if len(wandb_ids.keys()) > 2:
    #     ax = axs
    # else:
    #     ax = axs[i]
    ax = axs[i]
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 1 decimal place
    for j, (algo, id) in enumerate(wandb_ids.items()):
        if "PPO" in algo:
            stretch = 20
            means, stes = analyze_run(id, run_length)
            x = list(range(0, len(means), stretch))
            means = means[:len(x)]
        else:
            means, stes = analyze_offline_runs(id)#analyze_run(id)
            x = list(range(len(means)))
        if algo == "MAML+PPO":
            x, ppo_mean = run_means[name]["PPO"]
            means -= ppo_mean
        elif algo == "SAC (Pretrained)":
            x, sac_mean = run_means[name]["SAC (Vanilla)"]
            means -= sac_mean
        run_means[name][algo] = (x, means)
        # if len(wandb_ids.keys()) > 2:
        #     ax.errorbar(x, means, yerr = stes, label=algo, linewidth=3.0, color=colors[j])
        # else:
        #     ax.errorbar(x, means, yerr = stes, label=algo, linewidth=3.0)
        if name != "MAML+PPO vs Pretrained SAC" or (algo == "MAML+PPO" or algo == "SAC (Pretrained)"):
            ax.plot(x, means, label=algo, linewidth=3.0)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.set_title(name)
    ax.set_ylabel("Average Reward", fontsize=40)
    if True:#i == 0:
        ax.legend()
plt.xlabel("Environment Sampled Steps", fontsize=40)

fig.tight_layout()
plt.savefig("offline_experiments2.png")