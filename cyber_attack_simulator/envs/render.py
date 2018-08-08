import networkx as nx
import tkinter as Tk
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Viewer(object):
    """
    Displays sequence of observations from Cyber Attack Simulator Environment
    in a seperate window
    """

    def __init__(self, episode, network, width=7, height=7):
        self.episode = episode
        self.network = network
        self.width = width
        self.height = height
        # used for moving between timesteps in episode
        self.timestep = 0
        self.positions = self._get_machine_positions(episode[0][0])

        self.root = Tk.Tk()
        self.root.wm_title("Cyber Attack Simulator")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.quit)
        # matplotlib figure to house networkX graph
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.axes = self.fig.add_subplot(111)
        plt.tight_layout(pad=2)
        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        # buttons for moving between observations
        back = Tk.Button(self.root, text="back", command=self._previous_graph)
        back.pack()
        next = Tk.Button(self.root, text="next", command=self._next_graph)
        next.pack()
        # draw first observation
        self._next_graph()

        Tk.mainloop()

    def _next_graph(self):
        """
        Display next timestep in episode
        """
        if self.timestep < len(self.episode):
            G = self._get_graph(self.episode[self.timestep])
            self._draw_graph(G)
            self.timestep += 1

    def _previous_graph(self):
        """
        Display previous timestep in episode
        """
        if self.timestep > 1:
            self.timestep -= 2
            self._next_graph()

    def _get_graph(self, timestep):
        """
        Create a network graph from the current timestep

        Arguments:
            (State, Action, float) : tuple of state, action and reward from
                timestep within episode
        """
        G = nx.Graph()
        # sort nodes by their subnet
        subnets = {}
        state = timestep[0]
        for m in state.get_machines():
            if subnets.get(m[0]) is None:
                subnets[m[0]] = []
            subnets[m[0]].append(m)
        # Create a fully connected graph for each subnet
        for subnet in subnets.values():
            for m in subnet:
                node_color = self._machine_color(state, m)
                node_pos = self.positions[m]
                G.add_node(m, color=node_color, pos=node_pos)
            for x in subnet:
                for y in subnet:
                    if x == y:
                        continue
                    G.add_edge(x, y)

        # Create single edge between a single node from each subnet graph
        subnet_prime_nodes = []
        for subnet in subnets.values():
            subnet_prime_nodes.append(subnet[0])
        for x in subnet_prime_nodes:
            for y in subnet_prime_nodes:
                if x == y:
                    continue
                if self.network.subnets_connected(x[0], y[0]):
                    G.add_edge(x, y)
        return G

    def _get_machine_positions(self, state):
        """
        Get list of positions for each machine in episode
        """
        max = 100
        positions = {}
        machines = state.get_machines()
        nM = len(machines)
        margin = max / nM
        # calculate depth and width of network
        depth = np.floor(np.log2(np.ceil((nM - 2) / 5))) + 1
        row_height = max / depth
        col_max_width = max / 2**depth

        # positions are randomly assigned within regions of display based on
        # subnet number
        for m in machines:
            if m[0] == 0:
                # first subnet displayed in left half of first row
                row_min = max - row_height
                row_max = max
                col_min = 0
                col_max = max / 2.0
            elif m[0] == 1:
                # second subnet displayed in right half of first row
                row_min = max - row_height
                row_max = max
                col_min = max / 2.0
                col_max = max
            else:
                # all other subnets (user) displayed in tree
                subnet_depth = np.floor(np.log2(m[0] - 1))
                num_cols = 2**subnet_depth
                col_width = max / (num_cols)
                row = depth - (subnet_depth + 1)
                col = (m[0] - 1) - num_cols
                row_min = (row - 1) * row_height
                row_max = row * row_height
                col_min = col * col_width
                col_max = (col + 1) * col_width

            col_mid = col_max - ((col_max - col_min) / 2)
            row_pos = random.uniform(row_min + margin, row_max - margin)
            # narrow width so its even for all rows
            col_pos = random.uniform(col_mid - col_max_width + margin,
                                     col_mid + col_max_width - margin)
            positions[m] = (col_pos, row_pos)
        return positions

    def _draw_graph(self, G):
        """
        Draw the graph to the canvas
        """
        pos = {}
        colors = []
        for n in list(G.nodes):
            colors.append(G.nodes[n]["color"])
            pos[n] = G.nodes[n]["pos"]

        self.axes.cla()
        nx.draw_networkx_nodes(G, pos, node_color=colors, ax=self.axes)
        nx.draw_networkx_edges(G, pos)

        plt.axis('off')
        self._legend()
        state, action, reward, done = self.episode[self.timestep]
        if done:
            title = "t = {0}, Goal reached, total reward = {1}".format(
                self.timestep, reward)
        else:
            title = "t = {0}, {1}, reward = {2}".format(
                self.timestep, action, reward)
        self.axes.set_title(title)
        self.canvas.draw()

    def _legend(self):
        """
        Manually setup the display legend
        """
        s = mpatches.Patch(color='magenta', label='Sensitive (S)')
        c = mpatches.Patch(color='green', label='Compromised (C)')
        r = mpatches.Patch(color='blue', label='Reachable (R)')
        sc = mpatches.Patch(color='yellow', label='S & C')
        sr = mpatches.Patch(color='orange', label='S & R')
        o = mpatches.Patch(color='red', label='not S, C or R')

        legend_entries = [s, c, r, sc, sr, o]
        plt.legend(handles=legend_entries)

    def _machine_color(self, state, m):
        """
        Set machine color based on observation status
        """
        compromised = state.compromised(m)
        reachable = state.reachable(m)
        sensitive = state.sensitive(m)
        if sensitive:
            if compromised:
                color = 'yellow'
            elif reachable:
                color = 'orange'
            else:
                color = "magenta"
        elif compromised:
            color = 'green'
        elif reachable:
            color = 'blue'
        else:
            color = 'red'
        return color
