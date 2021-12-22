import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

def ma(data, window_width=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def read_results(PATH, TAG, name_func):
    df = pd.read_csv(PATH)
    col_names = df.columns
    groups = {}
    group_mins = {}
    group_maxs = {}
    for i, name in enumerate(col_names):
        if i == 0:
            # Skip step parameter
            continue
        grp, tag = name.split(' - ')
        parts = tag.strip().split('__')
        print(grp, tag)
        if TAG == tag.strip():
            groups[grp] = df.iloc[:, [0, i]]
            groups[grp] = groups[grp].dropna()
        if len(parts) > 1 and parts[0] == TAG:
            grp = name_func(grp)
            if parts[1] == 'MAX':
                group_maxs[grp] = df.iloc[:, [0, i]]
                group_maxs[grp] = group_maxs[grp].dropna()
            elif parts[1] == 'MIN':
                group_mins[grp] = df.iloc[:, [0, i]]
                group_mins[grp] = group_mins[grp].dropna()
            
    return groups, group_maxs, group_mins

def draw(PATH, TAG, x_label, y_label, data_dir="data/", yticks=None, xticks=None,linewidth=3, axis_width=3, format=".eps", figs_dir="figs/", name_func=lambda x:x, data_func=lambda x: x, fig_name="fig" , window_width=1, reset=True, name_filter=lambda x: True):
    if data_dir:
        PATH = os.path.join(data_dir, PATH)
    groups, group_maxs, group_mins = read_results(PATH, TAG, name_func)
    combined = list(groups.items())
    for i, (label, _) in enumerate(combined):
        label = name_func(label)
        if "Baseline" == label:
            combined = combined[i:i+1] + combined[:i] + combined[i+1:]
    for col_name, df in combined:
        
        if not name_filter(col_name):
            #Skip this one
            continue
        #print(df)
        name = name_func(col_name)

        ys = data_func(df.iloc[:, 1]).to_numpy()
        xs = df.iloc[:, 0].to_numpy()
        if window_width > 1:
            ys = ma(ys, window_width)
            xs = xs[window_width//2:-window_width//2+1]
        errs = np.zeros_like(xs)
        if name in group_maxs:
            max_df = group_maxs[name]
            max_ys = data_func(max_df.iloc[:, 1]).to_numpy()
            max_xs = max_df.iloc[:, 0].to_numpy()
            # if window_width > 1:
            #     max_ys = ma(max_ys, window_width)
            #     max_xs = max_xs[window_width//2:-window_width//2+1]
            min_df = group_mins[name]
            min_ys = data_func(min_df.iloc[:, 1]).to_numpy()
            min_xs = min_df.iloc[:, 0].to_numpy()
            errs = []
            for min_, max_ in zip(min_ys, max_ys):
                if max_ != min_:
                    errs.append((max_ - min_) / 2)
                elif len(errs) == 0:
                    errs.append(0)
                else:
                    errs.append(sum(errs[-5:])/len(errs[-5:]))
            errs = errs[window_width//2:-window_width//2+1]
            # if window_width > 1:
            #     errs = ma(errs, window_width)
                #min_xs = min_xs[window_width//2:-window_width//2+1]
            #plt.fill_between(max_xs, min_ys, max_ys, alpha=0.2)

        if name == "Baseline":
            #plt.errorbar(xs, ys, label=name, linewidth=linewidth, c='b', yerr=errs)
            p = plt.plot(xs, ys, label=name, linewidth=linewidth, c='b')
            
        else:
            #plt.errorbar(xs, ys, label=name, linewidth=linewidth, yerr = errs)
            p = plt.plot(xs, ys, label=name, linewidth=linewidth)
        plt.fill_between(xs[::1], (ys - errs)[::1], (ys + errs)[::1], alpha=0.3, interpolate=True, facecolor=p[-1].get_color())


    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(axis_width)
    plt.gca().spines['bottom'].set_linewidth(axis_width)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)
    plt.legend(fontsize=10, bbox_to_anchor=(0.95, 1.05))
    plt.tight_layout()
    save_fig_name = os.path.join(figs_dir, "{}.{}".format(fig_name, format))
    plt.savefig(save_fig_name, format=format, dpi=80)
    if reset:
        plt.clf()