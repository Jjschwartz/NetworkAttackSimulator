import networkx as nx
import matplotlib.pyplot as plt
import time


def machine_color(obs_m):
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


def graph(obs):
    """
    Render a network graph from the current observation

    Arguments:
        dict obs : ordered observation dictionary from CyberAttackSimulatorEnv
            class
    """
    G = nx.Graph()

    subnet_nodes = [[], [], []]
    for m in obs.keys():
        subnet_nodes[m[0]-1].append(m)

    # Create a fully connected graph for each subnet
    for i in range(len(subnet_nodes)):
        for m in subnet_nodes[i]:
            G.add_node(m, color=machine_color(obs[m]))
        for n in subnet_nodes[i]:
            for x in subnet_nodes[i]:
                if n == x:
                    continue
                G.add_edge(n, x)

    # Create a graph of subnet graphs
    subnet_prime_nodes = []
    for i in subnet_nodes:
        subnet_prime_nodes.append(i[0])
    for x in subnet_prime_nodes:
        for y in subnet_prime_nodes:
            if x == y:
                continue
            G.add_edge(x, y)

    return G


def draw_graph(G):
    plt.subplot(1, 1, 1)

    subnet_nodes = [[], [], []]
    pos = {}
    for n in list(G.nodes):
        subnet_nodes[n[0]-1].append(n)
        pos[n] = n

    for i in range(len(subnet_nodes)):
        for n in subnet_nodes[i]:
            nx.draw_networkx_nodes(G, pos, nodelist=[n],
                                   node_color=G.nodes[n]["color"])
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.ion()
    plt.show()


def wait_and_close(wait=2):
    time.sleep(wait)
    plt.close('all')
