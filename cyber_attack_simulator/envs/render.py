import networkx as nx
import tkinter as Tk
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     # noqa E402
import matplotlib.pyplot as plt         # noqa E402
import matplotlib.patches as mpatches   # noqa E402

# Agent node in graph
AGENT = (0, 0)

# Colors and symbols for describing state of machine
COLORS = ['yellow', 'orange', 'magenta', 'green', 'blue', 'red', 'black']
SYMBOLS = ['C', 'R', 'S', 'c', 'r', 'o', 'A']


class Viewer(object):
    """
    A class for visualizing the network state form Cyber Attack Simulator Environment
    """

    def __init__(self, network):
        """
        Initialize the Viewer, generating the network Graph given network topology and state

        Arguments:
            Network network : Network object describing network of environment
            State state : state of network user wants to view (Typically will be initial state)
        """
        self.network = network
        self.subnets = self._get_subnets(network)
        self.positions = self._get_machine_positions(network)

    def render_graph(self, state, ax=None, show=False):
        """
        Render graph structure representing network that can be then be visualized

        Arguments:
            State state : state of network user wants to view (Typically will be initial state)
            Axes ax : matplotlib axis to plot graph on, or None to plot on new axis
            bool show : whether to display plot, or simply construct plot
        """
        G = self._construct_graph(state)
        colors = []
        labels = {}
        for n in list(G.nodes):
            colors.append(G.nodes[n]["color"])
            labels[n] = G.nodes[n]["label"]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        nx.draw_networkx_nodes(G, self.positions, node_color=colors, ax=ax)
        nx.draw_networkx_labels(G, self.positions, labels, font_size=8)
        nx.draw_networkx_edges(G, self.positions)
        ax.axis('off')
        ax.set_xlim(left=0.0, right=100.0)

        legend_entries = EpisodeViewer.legend(compromised=False)
        ax.legend(handles=legend_entries)

        if show:
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    def render_episode(self, episode, width=7, height=7):
        """
        Display an episode from Cyber Attack Simulator Environment in a seperate window. Where an
        episode is a sequence of (state, action, reward, done) tuples generated from interactions
        with environment.

        Arguments:
            list episode : list of (State, Action, reward, done) tuples
            int width : width of GUI window
            int height : height of GUI window
        """
        init_ep_state = episode[0][0]
        G = self._construct_graph(init_ep_state)
        EpisodeViewer(episode, G, self.network.get_sensitive_machines(), width, height)

    def render_readable(self, state):
        """
        Print a readable version of state to stdout

        Arguments:
            State state : state of network user wants to view (Typically will be initial state)
        """
        output = ""
        for m in self.network.get_address_space():
            output += "Machine = " + str(m) + " =>\n"
            output += "\tServices:\n"
            for s in range(self.network.num_services):
                service_state = state.service_state(m, s)
                output += "\t\t{0} = {1}".format(s, str(service_state))
                output += "\n"
            output += "\treachable: {0}\n".format(state.reachable(m))
            output += "\tcompromised: {0}\n".format(state.compromised(m))
            output += "\tsensitive: {0}\n".format(self.network.is_sensitive_machine(m))
        print(output)

    def render_asci(self, state):
        """
        Render state in ASCI format to stdout

        Arguments:
            State state : state of network user wants to view (Typically will be initial state)
        """
        sensitive_machines = self.network.get_sensitive_machines()
        subnets = [[] for x in range(len(self.subnets))]
        for m in self.network.get_address_space():
            subnets[m[0]].append(get_machine_representation(state, sensitive_machines, m, SYMBOLS))

        max_row_size = max([len(x) for x in subnets])
        min_row_size = min([len(x) for x in subnets])

        output = "-----------------------------"
        for i, row in enumerate(subnets):
            output += "\nsubnet {0}: ".format(i)
            output += " " * ((max_row_size - len(row)) // 2)
            for col in row:
                output += col
            output += "\n"
            if i < len(subnets) - 1:
                n_spaces = (max_row_size - min_row_size) // 2
                output += " " * (len("subnet X: ") + n_spaces) + "|"
        output += "-----------------------------\n\n"

        print(output)

    def _construct_graph(self, state):
        """
        Create a network graph from the current state

        Arguments:
            State state : current state of network

        Returns:
            Graph G : NetworkX Graph representing state of network
        """
        G = nx.Graph()
        sensitive_machines = self.network.get_sensitive_machines()

        # Create a fully connected graph for each subnet
        for subnet in self.subnets:
            for m in subnet:
                node_color = get_machine_representation(state, sensitive_machines, m, COLORS)
                node_pos = self.positions[m]
                G.add_node(m, color=node_color, pos=node_pos, label=str(m))
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

        # positions are randomly assigned within regions of display based on subnet number
        positions = {}
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
        # add internet machine
        subnets[0].append(AGENT)
        return subnets


class EpisodeViewer(object):
    """
    Displays sequence of observations from Cyber Attack Simulator Environment in a seperate window
    """

    def __init__(self, episode, G, sensitive_machines, width=7, height=7):
        self.episode = episode
        self.G = G
        self.sensitive_machines = sensitive_machines
        # used for moving between timesteps in episode
        self.timestep = 0
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
        self.root.wm_protocol("WM_DELETE_WINDOW", self._close)
        # matplotlib figure to house networkX graph
        self.fig = plt.figure(figsize=(width, height))
        self.axes = self.fig.add_subplot(111)
        plt.tight_layout(pad=2)
        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        # buttons for moving between observations
        back = Tk.Button(self.root, text="back", command=self._previous_graph)
        back.pack()
        next = Tk.Button(self.root, text="next", command=self._next_graph)
        next.pack()

    def _close(self):
        plt.close('all')
        self.root.destroy()

    def _next_graph(self):
        """
        Display next timestep in episode
        """
        if self.timestep < len(self.episode):
            t_state = self.episode[self.timestep][0]
            self.G = self._update_graph(self.G, t_state)
            self._draw_graph(self.G)
            self.timestep += 1

    def _previous_graph(self):
        """
        Display previous timestep in episode
        """
        if self.timestep > 1:
            self.timestep -= 2
            self._next_graph()

    def _update_graph(self, G, state):
        """
        Update the graph G for a given state

        Arguments:
            State state : a state object

        Returns:
            Graph G : Updated NetworkX Graph representing state of network
        """
        # update colour of each machine in network as necessary
        for m in list(G.nodes):
            if m == AGENT:
                continue
            node_color = get_machine_representation(state, self.sensitive_machines, m, COLORS)
            G.nodes[m]["color"] = node_color
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

        # clear window and redraw graph
        self.axes.cla()
        nx.draw_networkx_nodes(G, pos, node_color=colors, ax=self.axes)
        nx.draw_networkx_edges(G, pos)
        plt.axis('off')
        # generate and plot legend
        legend_entries = self.legend()
        plt.legend(handles=legend_entries)
        # add title
        state, action, reward, done = self.episode[self.timestep]
        if done:
            title = "t = {0}, Goal reached, total reward = {1}".format(self.timestep, reward)
        else:
            title = "t = {0}, {1}, reward = {2}".format(self.timestep, action, reward)
        self.axes.set_title(title)
        self.canvas.draw()

    @staticmethod
    def legend(compromised=True):
        """
        Manually setup the display legend
        """
        a = mpatches.Patch(color='black', label='Agent')
        s = mpatches.Patch(color='magenta', label='Sensitive (S)')
        c = mpatches.Patch(color='green', label='Compromised (C)')
        r = mpatches.Patch(color='blue', label='Reachable (R)')
        legend_entries = [a, s, c, r]
        if compromised:
            sc = mpatches.Patch(color='yellow', label='S & C')
            sr = mpatches.Patch(color='orange', label='S & R')
            o = mpatches.Patch(color='red', label='not S, C or R')
            legend_entries.extend([sc, sr, o])
        return legend_entries


def get_machine_representation(state, sensitive_machines, m, representation):
    """
    Get the representation of a machine based on current state

    Arguments:
        State state : current state
        list sensitive_machines : list of addresses of sensitive machines on network
        (int, int) m : machine address
        list representation : list of different representations (e.g. color or symbol)

    Returns:
        str color : machine color
    """
    # agent not in state so return straight away
    if m == AGENT:
        return representation[6]
    compromised = state.compromised(m)
    reachable = state.reachable(m)
    sensitive = m in sensitive_machines
    if sensitive:
        if compromised:
            output = representation[0]
        elif reachable:
            output = representation[1]
        else:
            output = representation[2]
    elif compromised:
        output = representation[3]
    elif reachable:
        output = representation[4]
    else:
        output = representation[5]
    return output
