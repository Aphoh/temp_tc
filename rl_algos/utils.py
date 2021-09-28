import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ray.tune.logger import UnifiedLogger
import numpy as np
import datetime as dt

def fourier_points_from_action(action, points_length, fourier_basis_size):
    assert fourier_basis_size == (action.size + 1) // 2, "Incorrect fourier basis size for actions"
    root_points = (action[0]/2)*np.ones(points_length)
    inp = np.linspace(0, 1, points_length)
    for k in range(1, fourier_basis_size):
        ak, bk = action[2*k - 1], action[2*k]
        root_points += ak * np.sin(2*np.pi*k*inp) + bk * np.cos(2*np.pi*k*inp)

    points = root_points**2
    #TODO: More elegant solution than clipping
    points = np.clip(points, 0, 10)
    return points


def custom_logger_creator(log_path):

    def logger_creator(config):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return UnifiedLogger(config, log_path, loggers=None)

    return logger_creator

#Helper function
def string2bool(input_str: str):
    """
    Purpose: Convert strings to boolean. Specifically 'T' -> True, 'F' -> False
    """
    input_str = input_str.upper()
    if(input_str == 'T'):
        return True
    elif(input_str == 'F'):
        return False
    else:
        raise NotImplementedError ("Unknown boolean conversion for string: {}".format(input_str))



def plotter_person_reaction(data_dict, log_dir):

    print(data_dict)

    if len(data_dict) < 4:
        print("setup a method for less than four instances!")
        return

    if len(data_dict)==4:
        fig, axs = plt.subplots(2, 2, sharex = True, sharey=True)

        steps = list(data_dict.keys())
        steps.remove("control")

        ## control plot
        data = data_dict["control"]
        axs[0, 0].plot(data["x"], data["energy_consumption"])
        axs[0, 0].set_title('Control, rew: '+ str(round(data["reward"], 2)))
        axs[0, 0].set_xlabel("Time (hours)")
        axs[0, 0].set_ylabel("Energy Consumption (kWh)")
        # secondary axis for control
        axs002= axs[0, 0].twinx()
        axs002.set_ylabel("Dollars ($)", color = "red")
        axs002.plot(data["x"], data["grid_price"], color = "red")
        axs002.tick_params(axis="y", labelcolor = "red")

        ## Step 10
        data = data_dict[steps[0]]
        scaler = MinMaxScaler(feature_range = (0, 10))
        scaled_grid_price = scaler.fit_transform(np.array(data["grid_price"]).reshape(-1, 1))
        lns1 = axs[0, 1].plot(data["x"], data["energy_consumption"], label = "Energy")
        axs[0, 1].set_ylabel("Energy Consumption (kWh)")
        axs[0, 1].set_title(str(steps[0]) + " rew: " + str(round(data["reward"], 2)))
        axs[0, 1].set_xlabel("Time (hours)")
        # secondary axis for step 10
        axs002a = axs[0, 1].twinx()
        axs002a.set_ylabel("Points, prices scaled to points", color = "red")
        lns2 = axs002a.plot(data["x"], data["action"], color = "blue", label = "Agent")
        lns3 = axs002a.plot(data["x"], scaled_grid_price, color = "red", label = "Grid")
        axs002a.tick_params(axis="y", labelcolor = "red")
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        axs[0, 1].legend(lns, labs, loc=2)

        ## Step 1000
        data = data_dict[steps[1]]
        scaler = MinMaxScaler(feature_range = (0, 10))
        scaled_grid_price = scaler.fit_transform(np.array(data["grid_price"]).reshape(-1, 1))
        axs[1, 0].plot(data["x"], data["energy_consumption"])
        axs[1, 0].set_ylabel("Energy Consumption (kWh)")
        axs[1, 0].set_title(str(steps[1]) + " rew: " + str(round(data["reward"], 2)))
        axs[1, 0].set_xlabel("Time (hours)")
        # secondary axis for step 10
        axs002a= axs[1, 0].twinx()
        axs002a.set_ylabel("Points, prices scaled to points", color = "red")
        axs002a.plot(data["x"], data["action"], color = "blue")
        axs002a.plot(data["x"], scaled_grid_price, color = "red")
        axs002a.tick_params(axis="y", labelcolor = "red")

        ## Step 10000
        data = data_dict[steps[2]]
        scaler = MinMaxScaler(feature_range = (0, 10))
        scaled_grid_price = scaler.fit_transform(np.array(data["grid_price"]).reshape(-1, 1))
        axs[1, 1].plot(data["x"], data["energy_consumption"])
        axs[1, 1].set_ylabel("Energy Consumption (kWh)")
        axs[1, 1].set_title(str(steps[2]) + " rew: " + str(round(data["reward"])))
        axs[1, 1].set_xlabel("Time (hours)")
        # secondary axis for step 10
        axs002a= axs[1, 1].twinx()
        axs002a.set_ylabel("Points, prices scaled to points", color = "red")
        axs002a.plot(data["x"], data["action"], color = "blue")
        axs002a.plot(data["x"], scaled_grid_price, color = "red")
        axs002a.tick_params(axis="y", labelcolor = "red")

        # for ax in axs.flat:
        #     ax.set(xlabel='Time (hours)', ylabel="Energy Consumption (kWh)")

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()
        fig.tight_layout()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        outdir = os.path.join(log_dir, "reaction.pdf")
        print("Saving to", outdir)
        fig.savefig(outdir)
        return

def fourier_plotter_person_reaction(points_length, fourier_basis_size):
    def new_plotter_fn(data_dict, log_dir):
        # print(data_dict)
        new_ddict = {}
        for k, data in data_dict.items():
            dnew = data.copy()
            if "action" in dnew:
                dnew["action"] = fourier_points_from_action(data["action"], points_length, fourier_basis_size)
            new_ddict[k] = dnew

        # plotter_person_reaction(new_ddict, log_dir)

    return new_plotter_fn


