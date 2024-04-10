

import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
import time
import pandas as pd
import os

path_to_points = "./evaluated_points.txt"
path_to_results = "./results.txt"
path_to_exports = "./exports/optimization_results/"

if not os.path.exists(path_to_exports):
    os.mkdir(path_to_exports)

loop_limiter = True  # if true only only only one plot will be made

keep_plotting = True

manual_reference_error = 0.1886182005521773

_start_time = time.time()
_time_passed = 0

while (_time_passed < 200) and keep_plotting:

    points = np.genfromtxt(path_to_points)
    results = np.genfromtxt(path_to_results)
    # construct dataframe
    df = pd.DataFrame()

    for i in range(points.shape[1]):
        df[f"parameter {i+1}"] = points[:,i]
    df["results_raw"] = results

    # transform results
    results = results / manual_reference_error
    df["results"] = results
    df["baseline error raw"] = points.shape[0] * [manual_reference_error]
    df.to_csv(os.path.join(path_to_exports, "optimization_results.csv"))

    sorting_indizes = np.argsort(results)
    sorted_results = results[sorting_indizes]
    sorted_points = points[sorting_indizes]

    # compute delta
    points_delta =  points[1:, ] - points[0:-1, :]
    results_delta = results[1:] - results[0:-1]

    last_value = results[-1]
    best_value = np.min(results)
    point_of_best_value = points[np.argmin(results)]
    point_of_best_value_str = [f"{x:.3f}" for x in point_of_best_value]
    point_of_best_value_str = ", ".join(point_of_best_value_str)
    print(f"best value: {best_value:.4f}")
    print(f"scalar of best value: {point_of_best_value_str}")

    fig = plt.figure(figsize=(8,5))
    for dim in range(points.shape[1]):
        plt.plot(points[:,dim], label=f"scalar {dim}")
    plt.title(f"tested parameter settings\nparameters of best value: {point_of_best_value_str}")
    plt.xlabel("iteration")
    plt.ylabel("parameter value")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_exports, "tested parameter settings.png"), dpi=300)

    fig = plt.figure(figsize=(8,5))
    plt.plot(results)
    plt.title(f"results: last value: {last_value:.5f}, best value: {best_value:.5f}")
    plt.xlabel("iteration")
    plt.ylabel("results")
    plt.ylim((0.9*best_value, 1.2))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(path_to_exports, "results.png"), dpi=300)

    # delta
    fig = plt.figure(figsize=(8,5))
    for dim in range(points_delta.shape[1]):
        plt.plot(points_delta[:,dim], label=f"scalar {dim}")
    plt.title(f"change of tested parameter settings\nparameters of best value: {point_of_best_value_str}")
    plt.xlabel("iteration")
    plt.ylabel("parameter value i - (i-1)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_exports, "tested parameter settings_delta.png"), dpi=300)

    fig = plt.figure(figsize=(8,5))
    plt.plot(results_delta)
    plt.title(f"results delta: last value: {last_value:.5f}, best value: {best_value:.5f}")
    plt.xlabel("iteration")
    plt.ylabel("results i - (i-1)")
    plt.yscale("log")
    #plt.ylim((0.9*best_value, 0.3))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(path_to_exports, "results_delta.png"), dpi=300)

    #########################################
    cmap = mlp.colors.LinearSegmentedColormap.from_list("", ["green","yellow","red"])
    n_params = points.shape[1]
    sorted_results_for_plotting = np.clip(sorted_results[::-1], a_min=None, a_max=1.0)
    sorted_points_for_plotting =sorted_points[::-1]
    # ten best points
    sorted_results_for_plotting = np.clip(sorted_results[::-1], a_min=None, a_max=1.0)
    sorted_points_for_plotting =sorted_points[::-1]

    bounds = (np.min(points), np.max(points))
    fig, axes = plt.subplots(nrows=n_params, ncols=n_params, sharex=True, sharey=True,
                                    figsize=(10, 10))
    
    exclude_combination = []
    for i in range(n_params):
        for j in range(n_params):
            if (i,j) not in exclude_combination:
                map = axes[i, j].scatter(sorted_points_for_plotting[:,i],
                                    sorted_points_for_plotting[:,j],
                                    c=sorted_results_for_plotting,
                                    s=5.0,
                                    cmap=cmap)
                map = axes[i, j].scatter(sorted_points_for_plotting[:,i],
                    sorted_points_for_plotting[:,j],
                    c=sorted_results_for_plotting,
                    s=5.0,
                    cmap=cmap)
                #axes[i, j].set_title(f"dim {i+1} by {j+1}")
                axes[i, j].set_xlabel(f"dim {i+1}")
                axes[i, j].set_ylabel(f"dim {j+1}")  
                axes[i, j].set_ylim(bounds)
                exclude_combination.append((j,i))

    plt.suptitle(f"results of optimization\n(all results above 1.0 are clipped)")
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.colorbar(map, ax=axes.ravel().tolist(), label="relative error (w.r.t baseline)", extend="max")

    plt.savefig(os.path.join(path_to_exports, "optimization_scatterplot.png"), dpi=300)

    ####################################################
    # Histogram
    fig = plt.figure(figsize=(8,5))
    plt.hist(results, bins=100)
    plt.title(f"results delta: last value: {last_value:.5f}, best value: {best_value:.5f}")
    plt.xlabel("results")
    plt.ylabel("count")
    #plt.yscale("log")
    #plt.ylim((0.9*best_value, 0.3))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(path_to_exports, "results_histogram.png"), dpi=300)


    plt.close('all')
    print("saved plots")
    
    _current_time = time.time()
    _time_passed = _current_time - _start_time

    if loop_limiter:
        keep_plotting = False
    else:
        time.sleep(5)