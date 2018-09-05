import networkx as nx
import tkinter as Tk
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Agent node in graph
AGENT = (-1, -1)


class Viewer(object):
    """
    Displays sequence of observations from Cyber Attack Simulator Environment
    in a seperate window
    """

    def __init__(self, episode, network, width=7, height=7):
        self.episode = episode
        self.network = network
        # used for moving between timesteps in episode
        self.timestep = 0
        self.subnets = self._get_subnets(network)
        self.positions = self._get_machine_positions(network)
        self._setup_GUI(width, height)
        # draw first observation
        self._next_graph()
        # Initialize GUI drawing loop
        Tk.mainloop()

    def _setup_GUI(self, width, height):
        """
        Setup all the elements for the GUI for displaying the network graphs.

        Initializes object variables:k
            Tk root : the root window for GUI
            FigureCanvasTkAgg canvas : the canvas object to draw figure onto
            Figure fig : the figure that holds axes
            Axes axes : the matplotlib figure axes to draw onto
        """
        # The GUI root window
        self.root = Tk.Tk()
        self.root.wm_title("Cyber Attack Simulator")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.quit)
        # matplotlib figure to house networkX graph
        self.fig = plt.figure(figsize=(width, height))
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

    def _next_graph(self):
        """
        Display next timestep in episode
        """
        if self.timestep < len(self.episode):
            t_state = self.episode[self.timestep][0]
            G = self._get_graph(t_state)
            self._draw_graph(G)
            self.timestep += 1

    def _previous_graph(self):
        """
        Display previous timestep in episode
        """
        if self.timestep > 1:
            self.timestep -= 2
            self._next_graph()

    def _get_graph(self, state):
        """
        Create a network graph from the current state

        Arguments:
            State state : a state object

        Returns:
            Graph G : NetworkX Graph representing state of network
        """
        G = nx.Graph()

        # Create a fully connected graph for each subnet
        for subnet in self.subnets:
            for m in subnet:
                node_color = self._machine_color(state, m)
                node_pos = self.positions[m]
                G.add_node(m, color=node_color, pos=node_pos)
            for x in subnet:
                for y in subnet:
                    if x == y:
                        continue
                    G.add_edge(x, y)

        # Retrieve first machine in each subnet
        subnet_prime_nodes = []
        for subnet in self.subnets:
            subnet_prime_nodes.append(subnet[0])
        # Connect connected subnets by creating edge between first machine from each subnet
        for x in subnet_prime_nodes:
            for y in subnet_prime_nodes:
                if x == y:
                    continue
                if self.network.subnets_connected(x[0], y[0]):
                    G.add_edge(x, y)

        # Add agent node
        G.add_node(AGENT, color='black', pos=self.positions[AGENT])
        # Add edge between agent and first machine on each exposed subnet
        for x in subnet_prime_nodes:
            if self.network.subnet_exposed(x[0]):
                G.add_edge(x, AGENT)

        return G

    def _get_machine_positions(self, network):
        """
        Get list of positions for each machine in episode

        Arguments:
            Network network : network object describing network configuration of environment
                              episode was generated from
        """
        address_space = network.get_address_space()
        depths = network.get_subnet_depths()
        max_depth = max(depths)
        # list of lists where each list contains subnet_id of subnets with same depth
        subnets_by_depth = [[] for i in range(max_depth + 1)]
        for subnet_id, subnet_depth in enumerate(depths):
            subnets_by_depth[subnet_depth].append(subnet_id)

        # max value of position in figure
        max_pos = 100
        # for spacing between rows and columns and spread of nodes within subnet
        margin = 5
        row_height = max_pos / (max_depth + 1)

        positions = {}
        # positions are randomly assigned within regions of display based on subnet number
        for m in address_space:
            m_subnet = m[0]
            m_depth = depths[m_subnet]
            # row is dependent on depth of subnet
            row_max = max_pos - (m_depth * row_height)
            row_min = max_pos - ((m_depth + 1) * row_height)
            # col width is dependent on number of subnets at same depth
            num_cols = len(subnets_by_depth[m_depth])
            col_width = max_pos / num_cols
            # col of machine dependent on subnet_id relative to other subnets of same depth
            m_col = subnets_by_depth[m_depth].index(m_subnet)
            col_min = m_col * col_width
            col_max = (m_col + 1) * col_width
            # randomly sample position of machine within row and column of subnet
            col_mid = col_max - ((col_max - col_min) / 2)
            row_pos = random.uniform(row_min + margin, row_max - margin)
            col_pos = random.uniform(col_mid - margin, col_mid + margin)
            positions[m] = (col_pos, row_pos)

        # get position of agent, which is just right of machine first machine in network
        first_m_pos = positions[address_space[0]]
        agent_row = first_m_pos[1]
        agent_col = min(first_m_pos[0] + margin * 4, max_pos - margin)
        positions[AGENT] = (agent_col, agent_row)

        return positions

    def _get_subnets(self, network):
        """
        Get list of machines organized into subnets

        Arguments:
            int num_subnets : number of subnets on network
            list[(int, int)] address_space : list of machine addresses on network

        Returns:
            list[list[(int, int)]] : list of lists of addresses with each list containing machines
                                     on same subnet
        """
        subnets = [[] for i in range(network.get_number_of_subnets())]
        for m in network.get_address_space():
            subnets[m[0]].append(m)
        return subnets

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
        a = mpatches.Patch(color='black', label='Agent')
        s = mpatches.Patch(color='magenta', label='Sensitive (S)')
        c = mpatches.Patch(color='green', label='Compromised (C)')
        r = mpatches.Patch(color='blue', label='Reachable (R)')
        sc = mpatches.Patch(color='yellow', label='S & C')
        sr = mpatches.Patch(color='orange', label='S & R')
        o = mpatches.Patch(color='red', label='not S, C or R')

        legend_entries = [a, s, c, r, sc, sr, o]
        plt.legend(handles=legend_entries)

    def _machine_color(self, state, m):
        """
        Set machine color based on observation status
        """
        compromised = state.compromised(m)
        reachable = state.reachable(m)
        sensitive = self.network.is_sensitive_machine(m)
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
