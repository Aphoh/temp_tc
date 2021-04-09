import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
runs = {
    "Adaptation to Threshold Response": {
        "MAML+PPO": "social-game-rl/energy-demand-response-game/gjg5p7lo",
        "PPO": "social-game-rl/energy-demand-response-game/3skaelp1"
        
    },
    "Adaptation to Curtail and Shift Response": {
        "MAML+PPO": "social-game-rl/energy-demand-response-game/2g90nma7",
        "PPO": "social-game-rl/energy-demand-response-game/14x4i8so"
    },
    "MAML+PPO Response to Number of Simulation Training Iterations": {
        "50 iterations": "social-game-rl/energy-demand-response-game/24nd2dj1",
        "100 iterations": "social-game-rl/energy-demand-response-game/2g90nma7",
        "150 iterations": "social-game-rl/energy-demand-response-game/3tc4ni7u",
        "200 iterations": "social-game-rl/energy-demand-response-game/2ev8r466"
    }
}
fig, axs = plt.subplots(len(runs.keys()), sharex=True, figsize=(20, 20))
for i, (name, wandb_ids) in enumerate(runs.items()):
    for algo, id in wandb_ids.items():
        means, stes = analyze_run(id)
        x = list(range(len(means)))
        axs[i].errorbar(x, means, yerr = stes, label=algo)
    axs[i].set_title(name)
    plt.ylabel("Average Reward")
plt.xlabel("Gradient Update Steps")
axs[0].legend()
axs[-1].legend()
fig.tight_layout()
plt.savefig("maml_summary_analysis_grid.png")