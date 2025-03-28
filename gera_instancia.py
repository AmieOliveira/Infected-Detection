"""
    Script to generate an instance for the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np
import networkx as nx
import epidemic
import centrality


parser = argparse.ArgumentParser(
    prog="Hidden network epidemics generator",
    description="Script designed to generate an instance of a network \n"
                "epidemic with hidden infected nodes. Customizations \n"
                "should be given via a configuration file."
)

parser.add_argument(
    "configfile",
    default="gera_instancia_config.yaml", nargs="?",
    help="Name of the configuration file to be used. If none is \n"
         "given, then the default 'gera_instancia_config.yaml' \n"
         "will be used."
)
parser.add_argument(
    "--seed", "-s", type=int,
    help="Random seed to be used in the data generation."
)

net_parser = parser.add_argument_group(
    "Network Arguments",
    description="Arguments to define the network configurations. Whatever \n"
                "arguments given will overwrite the parameters provided \n"
                "in the configuration file ",
)
net_parser.add_argument(
    "--model-type", "--model",
    choices=["BA", "WS", "ER", "adjacency"],
    help="Type of network model to be used. There are four possible \n"
         "models implemented: 'BA', for the Barabási-Albert model; \n"
         "'WS', for the Watts-Strogatz or small world model; 'ER', \n"
         "for the Erdös-Rényi or G(n,p) model; and 'adjacency', for \n"
         "when a file with the network adjacency matrix should be \n"
         "provided in place of generating a random model."
)
net_parser.add_argument(
    "--n-nodes", "-n", type=int,
    help="The number of nodes the random model should have."
)
net_parser.add_argument(
    "--BA-m", "-m", type=int,
    help="Barabási-Albert model parameter that gibes the number of \n"
         "edges a new node will have when it enters the network."
)
net_parser.add_argument(
    "-p", "--WS-p", "--ER-p", type=float,
    help="This parameter can be used for two different models. It \n"
         "may be used as the Watts-Strogatz model parameter that \n"
         "gives the probability of rewiring each edge. Or it may \n"
         "correspond to the Erdös-Rényi model for the probability \n"
         "of edge creation."
)
net_parser.add_argument(
    "--WS-k", "-k", type=int,
    help="Watts-Strogatz model parameter that gives the number of \n"
         "initial edges for each node (joined to the k nearest)."
)
net_parser.add_argument(
    "--path", "--adjacency-path",
    help="Path of the file with the provided network (used when \n"
         "the 'adjacency' model type is chosen)."
)

epi_parser = parser.add_argument_group(
    "Epidemic Arguments",
    description="Arguments to define the epidemic configurations. Whatever \n"
                "arguments given will overwrite the parameters provided \n"
                "in the configuration file ",
)
epi_parser.add_argument(
    "--beta", "-b", type=float,
    help="Probability of an infected node infecting an adjacent \n"
         "susceptible one"
)
epi_parser.add_argument(
    "--initial_infection", "--init", type=int,
    help="Number of initial infected nodes"
)
epi_parser.add_argument(
    "--time", "--epidemic-time", "-t", type=int,
    help="Total time of simulation (set to 0 to disable)"
)
epi_parser.add_argument(
    "--stop", "--strop-fraction", "-f", type=float,
    help="Maximum fraction of infected nodes before simulation \n"
         "stops (set to 0 to disable)"
)
epi_parser.add_argument(
    "--observ", "--observ-prob", "-o", type=float,
    help="Probability of an infected node being identified as \n"
         "infected"
)

# 1. Read from configuration file
args = parser.parse_args()
config_file = args.configfile
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

seed = args.seed
if not seed:
    seed = cfg["rd_seed"]
np.random.seed(seed)  # TODO: Conferir se isso é o suficiente

# 2. Create graph
net_model = args.model_type
if not net_model:
    net_model = cfg["network"]["model"]

net = None
if net_model == "adjacency":
    net_path = args.path
    if not net_path:
        net_path = cfg["network"]["adjacency"]["path"]
    G = nx.from_numpy_matrix(np.array(net_path))
else:
    n = args.n_nodes
    if not n:
        n = cfg["network"]["n_nodes"]

    if net_model == "BA":
        m = args.BA_m
        if not m:
            m = cfg["network"]["BA"]["m"]
        G = nx.barabasi_albert_graph(n, m)
    elif net_model == "WS":
        k = args.WS_k
        if not k:
            k = cfg["network"]["WS"]["k"]
        p = args.WS_p
        if not p:
            p = cfg["network"]["WS"]["p"]
        G = nx.watts_strogatz_graph(n, k, p)
    elif net_model == "ER":
        p = args.WS_p
        if not p:
            p = cfg["network"]["ER"]["p"]
        G = nx.erdos_renyi_graph(n, p)

# 3. Create epidemic
beta = args.beta if args.beta else cfg["epidemic"]["beta"]
init_infec = args.initial_infection if args.initial_infection else cfg["epidemic"]["init_infec"]
total_time = args.time if args.time else cfg["epidemic"]["total_time"]
stop_frac = args.stop if args.stop else cfg["epidemic"]["stop_frac"]
observ_prob = args.observ if args.observ else cfg["epidemic"]["observ_prob"]

infected_nodes = epidemic.si_epidemic(G, beta, init_infec, total_time, stop_frac)

observ_infec = epidemic.observed_infected(infected_nodes, observ_prob)


# 4. Compute network centralities
degree = centrality.degree(G)
contact = centrality.contact(G, observ_infec)
betweenness = centrality.betweenness(G)
observ_betweenness = centrality.observed_betweenness(G, observ_infec)


# 5. Format and save data
# TODO: Salvar dados (usando a classe, e definir nome de arquivo)

