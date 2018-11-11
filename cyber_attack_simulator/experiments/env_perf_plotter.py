"""
Generates plots for env_perf_exp

1. Actions per second vs M vs S
"""
from mpl_toolkits.mplot3d import Axes3D     # noqa F401
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import sys


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data_over_runs(df):
    avg_df = df.groupby(["M", "S"]).mean().reset_index()
    return avg_df


def plot_2D(df, fig, y_var, y_label):
    """ y_var vs M and y_var vs S """

    # scaling vs machines for 3 different service values
    ax1 = fig.add_subplot(121)
    S_all_values = df.S.unique()
    mid = (len(S_all_values)) // 2
    S_values = [S_all_values[0],
                # S_all_values[mid // 2],
                S_all_values[mid],
                # S_all_values[mid + mid // 2],
                S_all_values[-1]]
    for s_val in S_values:
        m_df = df.loc[df.S == s_val]
        avg_df = average_data_over_runs(m_df)
        Y = avg_df[y_var]
        err = m_df[y_var].sem()
        X = avg_df.M
        ax1.plot(X, Y, label=s_val)
        ax1.fill_between(X, Y-err, Y+err, alpha=0.3)

    m_avg_all = df.groupby(["M"]).mean().reset_index()
    Y = m_avg_all[y_var]
    err = df[y_var].sem()
    X = m_avg_all.M
    ax1.plot(X, Y, label="All")
    ax1.fill_between(X, Y-err, Y+err, alpha=0.3)

    ax1.set_xlabel("Machines")
    ax1.set_ylabel(y_label)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.locator_params(axis="x", nbins=4)
    ax1.locator_params(axis="y", nbins=4)
    ax1.legend(title="Services")

    # scaling vs servies for 3 different service values
    ax2 = fig.add_subplot(122)
    M_all_values = df.M.unique()
    mid = (len(M_all_values)) // 2
    M_values = [M_all_values[0],
                # M_all_values[mid // 2],
                M_all_values[mid],
                # M_all_values[mid + mid // 2],
                M_all_values[-1]]
    # M_values = [df.M.min(), int(df.M.median()), df.M.max()]
    for m_val in M_values:
        s_df = df.loc[df.M == m_val]
        avg_df = average_data_over_runs(s_df)
        Y = avg_df[y_var]
        err = s_df[y_var].sem()
        X = avg_df.S
        ax2.plot(X, Y, label=m_val)
        ax2.fill_between(X, Y-err, Y+err, alpha=0.3)

    s_avg_all = df.groupby(["S"]).mean().reset_index()
    Y = s_avg_all[y_var]
    err = df[y_var].sem()
    X = s_avg_all.S
    ax2.plot(X, Y, label="All")
    ax2.fill_between(X, Y-err, Y+err, alpha=0.3)

    ax2.set_xlabel("Services")
    ax2.set_ylabel(y_label)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.locator_params(axis="x", nbins=4)
    ax2.locator_params(axis="y", nbins=4)
    ax2.legend(title="Machines")

    fig.tight_layout()


def plot_action_per_sec_2D(df, fig):
    """ actions per sec vs M and actions per sec vs S """
    plot_2D(df, fig, "a_per_sec", "Mean actions per second")


def plot_time_per_action_2D(df, fig):
    """ time per action vs M and time per action vs S """
    plot_2D(df, fig, "t_per_a", "Mean time per action (sec)")


def plot_load_time_2D(df, fig):
    """ load time vs M and load time vs S """
    plot_2D(df, fig, "load_time", "Mean load time (sec)")


def plot_3D(df, ax, z_var, z_label):
    """ actions per sec vs M and S """
    df = average_data_over_runs(df)
    X = df.M
    Y = df.S
    Z = df[z_var]

    ax.plot_trisurf(X, Y, Z,  cmap='viridis')

    ax.set_xlabel("Machines", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.locator_params(axis="x", nbins=4)
    ax.set_ylabel("Services", fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.locator_params(axis="y", nbins=4)
    ax.set_zlabel(z_label, fontsize=12)
    ax.locator_params(axis="z", nbins=4)


def summary_stats(df, var):
    """ Print max and min values with sd dev """
    avg_df = average_data_over_runs(df)
    max_row = avg_df.loc[avg_df[var].idxmax()]
    min_row = avg_df.loc[avg_df[var].idxmin()]

    max_err = df[var].loc[(df["M"] == max_row["M"]) & (df["S"] == max_row["S"])].std()
    min_err = df[var].loc[(df["M"] == min_row["M"]) & (df["S"] == min_row["S"])].std()

    print("\nSummary stats for", var)
    print("max =\n", max_row)
    print("max stdev for {} = {}".format(var, max_err))
    print("\nmin =\n", min_row)
    print("min stdev for {} = {}".format(var, min_err))


def main():

    if len(sys.argv) != 2:
        print("Usage: python scaling_plotter.py <result_file>.csv")
        return 1

    print("Watch as it grows and grows and grows!")
    results_df = import_data(sys.argv[1])

    summary_stats(results_df, "load_time")
    summary_stats(results_df, "a_per_sec")
    summary_stats(results_df, "t_per_a")

    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111, projection='3d')
    # # plot_3D(results_df, ax, "load_time", "Mean Load time (sec)")
    # # plot_3D(results_df, ax, "a_per_sec", "Mean actions per second")
    # plot_3D(results_df, ax, "t_per_a", "Mean time per action (sec)")
    # fig1.tight_layout()
    #
    # fig2 = plt.figure(2)
    # plot_action_per_sec_2D(results_df, fig2)

    # fig3 = plt.figure(3)
    # plot_time_per_action_2D(results_df, fig3)

    # fig4 = plt.figure(4)
    # plot_load_time_2D(results_df, fig4)

    plt.show()


if __name__ == "__main__":
    main()
