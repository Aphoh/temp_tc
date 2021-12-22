import matplotlib.pyplot as plt
import matplotlib
from graph_logs import *
import os

def graph_nn_quantile():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=2
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(100 * float(num) / 20))
        return "quantile {}%".format(num)

    data_func = lambda x: (x+40) * 200
    draw("quantile_1000_nn.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_1000_nn",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)
    draw("baseline_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_1000_nn_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "Baseline", 
        window_width=5, 
        reset=False,
        data_func=data_func)
    draw("quantile_1000_nn_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_1000_nn_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        reset=True,
        data_func=data_func)
    draw("baseline_multi.csv", 
        TAG, 
        "Total Steps Collected (Planning+Real)", 
        "Annual Cost ($)", 
        fig_name="quantile_totalsteps_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "Baseline", 
        window_width=5, 
        reset=False,
        data_func=data_func)
    draw("quantile_totalsteps_multi.csv", 
        TAG, 
        "Total Steps Collected (Planning+Real)", 
        "Annual Cost ($)", 
        fig_name="quantile_totalsteps_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        reset=True,
        data_func=data_func)
    draw("quantile_500_nn.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_500_nn",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5,
        data_func=data_func)
    draw("quantile_10000_nn.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_10000_nn",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)

def graph_quantile_singlecomp():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=2
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Single Model"
        num = str(int(100 * float(num) / 20))
        return "Ensemble"

    data_func = lambda x: (x+40) * 200
    draw("baseline_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_singlecomp_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "Baseline", 
        window_width=5, 
        reset=False,
        data_func=data_func)
    draw("quantile_singlecomp_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_singlecomp_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        reset=True,
        data_func=data_func)

def graph_nn_quantile_fracreal():
    TAG="ray/tune/custom_metrics/is_step_in_real_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=2
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(100 * float(num) / 20))
        return "quantile {}%".format(num)

    data_func = lambda x: x
    draw("quantile_1000_fracreal.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Fraction of Time Spent in Real Environment", 
        fig_name="quantile_1000_fracreal",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=10, 
        data_func=data_func)
    draw("quantile_fracreal_multi.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Fraction of Time Spent in Real Environment", 
        fig_name="quantile_fracreal_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=10, 
        data_func=data_func)

def graph_nn_fracreal():
    TAG="ray/tune/custom_metrics/is_step_in_real_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(num))
        return "Dataset Size: {}".format(num)

    data_func = lambda x: x
    draw("nn_fracreal.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Fraction of Time Spent in Real Environment", 
        fig_name="nn_fracreal",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)

def graph_nn(plot_baseline=True, figname="nn"):
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=2
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(num))
        return "Dataset Size: {}".format(num)
    def name_filter(name):
        throwaway, num = name.split(':')
        return num.strip().isdigit()
    if plot_baseline:
        name_filter = lambda x: True
    data_func = lambda x: (x+40) * 200
    draw("nn.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name=figname,
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        name_filter=name_filter,
        data_func=data_func)
    draw("baseline_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_all_nn_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "Baseline", 
        window_width=5, 
        reset=False,
        data_func=data_func)
    draw("quantile_all_nn_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="quantile_all_nn_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=10, 
        reset=True,
        data_func=data_func)

def graph_nn_cross():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=2
    axis_width=3
    name_filter = lambda name: True
    name_func = lambda name: "Baseline" if "vanilla" in name else "Threshold Type: Hard"
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    data_func = lambda x: (x+40) * 200
    draw("cross_nn2.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_nn",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func, 
        reset=False, 
        name_filter=name_filter)
    def name_func(name):
        throwaway, num = name.split(':')
        num = num.strip()
        if num.strip().isdigit():
            return "Threshold Type: quantile ({}%)".format(int(num) * 100/20)
        return "Threshold Type: {}".format(num)

    name_filter = lambda name: "hard" not in name

    
    draw("cross_nn.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_nn",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func, 
        reset=True, 
        name_filter=name_filter)
    draw("baseline_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_nn_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "Baseline", 
        window_width=5, 
        reset=False,
        data_func=data_func)
    draw("cross_nn_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_nn_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=10, 
        reset=True,
        data_func=data_func)


def graph_nn_error():
    TAG="ray/tune/custom_metrics/prediction_error_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        return "Threshold Type = {}".format(num)

    data_func = lambda x: x * 100
    draw("cross_nn_error.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Prediction % Error", 
        fig_name="cross_nn_error",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)
def graph_quantile_error():
    TAG="ray/tune/custom_metrics/prediction_error_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(100 * float(num) / 20))
        return "quantile {}%".format(num)

    data_func = lambda x: x * 100
    draw("quantile_error_multi.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Prediction % Error", 
        fig_name="quantile_error_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)


def graph_oracle_cross_nn():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(num))
        return "Noise Std: {}".format(num)

    data_func = lambda x: (x+40) * 200
    draw("oracle_small.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_oracle_nn",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func, reset=False)
    graph_nn(plot_baseline=False, figname="cross_oracle_nn")

def graph_oracle():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if not num.strip().isdigit():
            return "Baseline"
        num = str(int(num))
        return "Noise Std: {}".format(num)

    data_func = lambda x: (x+40) * 200
    draw("oracle.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="oracle",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)

def graph_oracle_cross():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    name_filter = lambda name: True
    name_func = lambda name: "Baseline" if "vanilla" in name else "Threshold Type: Hard"
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    data_func = lambda x: (x+40) * 200
    draw("cross_oracle2.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_oracle",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func, 
        reset=False, 
        name_filter=name_filter)
    def name_func(name):
        throwaway, num = name.split(':')
        return "Threshold Type: {}".format(num)

    name_filter = lambda name: "hard" not in name
    
    draw("cross_oracle.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_oracle",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func, 
        reset=True, 
        name_filter=name_filter)
    draw("baseline_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_oracle_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=lambda x: "Baseline", 
        window_width=5, 
        reset=False,
        data_func=data_func)
    draw("cross_oracle_multi.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="cross_oracle_multi",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=10, 
        reset=True,
        data_func=data_func)

def graph_oracle_s():
    TAG="ray/tune/custom_metrics/energy_cost_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        if int(num) == 1:
            return "Baseline"
        num = str(int(num))
        return "Threshold Offset: {}".format(num)

    data_func = lambda x: (x+40) * 200
    draw("oracle_s.csv", 
        TAG, 
        "Real Steps Collected", 
        "Annual Cost ($)", 
        fig_name="oracle_s",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)

def graph_cross_oracle_error():
    TAG="ray/tune/custom_metrics/prediction_error_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        return "Noise Std = {}".format(num)

    data_func = lambda x: x * 100
    draw("cross_oracle_error.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Prediction % Error", 
        fig_name="cross_oracle_error",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)

def graph_oracle_error():
    TAG="ray/tune/custom_metrics/prediction_error_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        return "Noise Std = {}".format(num)

    data_func = lambda x: x * 100
    draw("oracle_error.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Prediction % Error", 
        fig_name="oracle_error",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)

def graph_oracle_fracreal():
    TAG="ray/tune/custom_metrics/is_step_in_real_mean"
    FIGS_DIR = "figs/"
    FORMAT='pdf'
    linewidth=3
    axis_width=3
    def name_func(name):
        throwaway, num = name.split(':')
        return "Noise Std = {}".format(num)

    data_func = lambda x: x
    draw("oracle_fracreal.csv", 
        TAG, 
        "Total Steps Collected (Real+Planning)", 
        "Fraction of Time Spent in Real Environment", 
        fig_name="oracle_fracreal",
        figs_dir=FIGS_DIR, 
        format=FORMAT, 
        linewidth=linewidth, 
        axis_width=axis_width, 
        name_func=name_func, 
        window_width=5, 
        data_func=data_func)


graph_nn_quantile()
graph_quantile_singlecomp()
graph_quantile_error()
graph_nn_quantile_fracreal()
graph_nn()
graph_nn_fracreal()
graph_nn_cross()
graph_nn_error()
graph_oracle_cross_nn()
graph_oracle()
graph_oracle_s()
graph_oracle_cross()
graph_cross_oracle_error()
graph_oracle_error()
graph_oracle_fracreal()
