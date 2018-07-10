import networkx as nx
import tkinter as Tk
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

    def __init__(self, obs, width=7, height=7):
        self.obs = obs
        self.width = width
        self.height = height
        # used for moving between observations in sequence
        self.obs_num = 0

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
        Display next observation in sequence
        """
        if self.obs_num < len(self.obs):
            G = self._get_graph(self.obs[self.obs_num])
            self._draw_graph(G)
            self.obs_num += 1

    def _previous_graph(self):
        """
        Display previous observation in sequence
        """
        if self.obs_num > 1:
            self.obs_num -= 2
            self._next_graph()

    def _get_graph(self, obs):
        """
        Create a network graph from the current observation
        """
        G = nx.Graph()
        # sort nodes by their subnet
        subnets = [[], [], []]
        for m in obs.keys():
            subnets[m[0]-1].append(m)
        # Create a fully connected graph for each subnet
        for subnet in subnets:
            node_y_step = 1.0 / (len(subnet) + 1)
            for m in subnet:
                node_color = self._machine_color(obs[m])
                node_pos = (m[0], (m[1] + 1) * node_y_step)
                G.add_node(m, color=node_color, pos=node_pos)
            for x in subnet:
                for y in subnet:
                    if x == y:
                        continue
                    G.add_edge(x, y)

        # Create single edge between a single node from each subnet graph
        subnet_prime_nodes = []
        for subnet in subnets:
            subnet_prime_nodes.append(subnet[0])
        for x in subnet_prime_nodes:
            for y in subnet_prime_nodes:
                if x == y:
                    continue
                G.add_edge(x, y)
        return G

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
        title = "t = {0}".format(self.obs_num)
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

    def _machine_color(self, obs_m):
        """
        Set machine color based on observation status
        """
        compromised = obs_m["compromised"]
        reachable = obs_m["reachable"]
        sensitive = obs_m["sensitive"]
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
