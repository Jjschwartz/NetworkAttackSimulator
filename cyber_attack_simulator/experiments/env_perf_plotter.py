"""
Generates plots for env_perf_exp

1. Actions per second vs M vs S
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def plot_action_per_sec(df, ax):

    X = df.M
    Y = df.S
    Z = df.a_per_sec

    Xv, Yv = np.meshgrid(X, Y)

    ax.plot_trisurf(X, Y, Z)

    ax.set_xlabel("Machines")
    ax.set_xticks(X)
    ax.set_ylabel("Services")
    ax.set_yticks(Y)
    ax.set_zlabel("Actions per second")


def main():

    if len(sys.argv) != 2:
        print("Usage: python scaling_plotter.py <result_file>.csv")
        return 1

    print("Watch as it grows and grows and grows!")
    results_df = import_data(sys.argv[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_action_per_sec(results_df, ax)

    plt.show()


if __name__ == "__main__":
    main()
