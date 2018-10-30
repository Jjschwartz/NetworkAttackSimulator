"""
Generates plots for env_perf_exp

1. Actions per second vs M vs S
"""
from mpl_toolkits.mplot3d import Axes3D     # noqa F401
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import sys


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data_over_runs(df):
    avg_df = df.groupby(["M", "S"]).mean().reset_index()
    return avg_df


def plot_action_per_sec_2D(df, fig):
    """ param = M or S """

    # scaling vs machines for 3 different service values
    ax1 = fig.add_subplot(121)
    S_values = [df.S.min(), int(df.S.median()), df.S.max()]
    for s_val in S_values:
        m_df = df.loc[df.S == s_val]
        avg_df = average_data_over_runs(m_df)
        Y = avg_df.a_per_sec
        err = m_df.a_per_sec.sem()
        X = avg_df.M
        ax1.plot(X, Y, label=s_val)
        ax1.fill_between(X, Y-err, Y+err, alpha=0.3)

    ax1.set_xlabel("Machines")
    ax1.set_ylabel("Mean actions per second")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(title="Services")

    # scaling vs servies for 3 different service values
    ax2 = fig.add_subplot(122)
    M_values = [df.M.min(), int(df.M.median()), df.M.max()]
    for m_val in M_values:
        s_df = df.loc[df.M == m_val]
        avg_df = average_data_over_runs(s_df)
        Y = avg_df.a_per_sec
        err = s_df.a_per_sec.sem()
        X = avg_df.S
        ax2.plot(X, Y, label=m_val)
        ax2.fill_between(X, Y-err, Y+err, alpha=0.3)

    ax2.set_xlabel("Services")
    ax2.set_ylabel("Mean actions per second")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(title="Machines")

    fig.tight_layout()


def plot_load_time_2D(df, fig):
    """ param = M or S """

    # scaling vs machines for 3 different service values
    ax1 = fig.add_subplot(121)
    S_values = [df.S.min(), int(df.S.median()), df.S.max()]
    for s_val in S_values:
        m_df = df.loc[df.S == s_val]
        avg_df = average_data_over_runs(m_df)
        Y = avg_df.load_time
        err = m_df.load_time.sem()
        X = avg_df.M
        ax1.plot(X, Y, label=s_val)
        ax1.fill_between(X, Y-err, Y+err, alpha=0.3)

    ax1.set_xlabel("Machines")
    ax1.set_ylabel("Mean load time (sec)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(title="Services")

    # scaling vs servies for 3 different service values
    ax2 = fig.add_subplot(122)
    M_values = [df.M.min(), int(df.M.median()), df.M.max()]
    for m_val in M_values:
        s_df = df.loc[df.M == m_val]
        avg_df = average_data_over_runs(s_df)
        Y = avg_df.load_time
        err = s_df.load_time.sem()
        X = avg_df.S
        ax2.plot(X, Y, label=m_val)
        ax2.fill_between(X, Y-err, Y+err, alpha=0.3)

    ax2.set_xlabel("Services")
    ax2.set_ylabel("Mean load time (sec)")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(title="Machines")

    fig.tight_layout()


def plot_action_per_sec_3D(df, ax):

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

    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111, projection='3d')
    # plot_action_per_sec_3D(results_df, ax)

    fig2 = plt.figure(2)
    plot_action_per_sec_2D(results_df, fig2)

    fig3 = plt.figure(3)
    plot_load_time_2D(results_df, fig3)

    plt.show()


if __name__ == "__main__":
    main()
