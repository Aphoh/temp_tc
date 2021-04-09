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
    num_runs = 50
    first_indices = []
    trial_names = []
    for i in range(50):
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
ppo_run = "social-game-rl/energy-demand-response-game/3skaelp1"
maml_run = "social-game-rl/energy-demand-response-game/gjg5p7lo"
ppo_means, ppo_stds = analyze_run(ppo_run)
maml_means, maml_stds = analyze_run(maml_run)
x = list(range(len(ppo_means)))
plt.errorbar(x, ppo_means, yerr = ppo_stds, label = "PPO")
plt.errorbar(x, maml_means, yerr = maml_stds, label = "MAML + PPO")
plt.legend()
plt.savefig("maml_summary_analysis_deterministicfn.png")