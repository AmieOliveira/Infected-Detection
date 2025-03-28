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
    description="Script designed to generate an instance of a network "
                "epidemic with hidden infected nodes. Customizations "
                "should be given via a configuration file."
)
parser.add_argument(
    "configfile",
    default="gera_instancia_config.yaml", nargs="?",
    help="Name of the configuration file to be used. "
         "If none is given, then the default "
         "'gera_instancia_config.yaml' will be used."
)


# 1. Read from configuration file
args = parser.parse_args()
config_file = args.configfile
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

seed = cfg["rd_seed"]
np.random.seed(seed)  # TODO: Conferir se isso é o suficiente

# 2. Create graph
net_model = cfg["network"]["model"]

net = None
if net_model == "adjacency":
    net_path = cfg["network"]["adjacency"]["path"]
    G = nx.from_numpy_matrix(np.array(net_path))
else:
    n = cfg["network"]["n_nodes"]
    if net_model == "BA":
        m = cfg["network"]["BA"]["m"]
        G = nx.barabasi_albert_graph(n, m)
    elif net_model == "WS":
        k = cfg["network"]["WS"]["k"]
        p = cfg["network"]["WS"]["p"]
        G = nx.watts_strogatz_graph(n, k, p)
    elif net_model == "ER":
        p = cfg["network"]["ER"]["p"]
        G = nx.erdos_renyi_graph(n, p)

# 3. Create epidemic
beta = cfg["epidemic"]["beta"]
init_infec = cfg["epidemic"]["init_infec"]
total_time = cfg["epidemic"]["total_time"]
stop_frac = cfg["epidemic"]["stop_frac"]
observ_prob = cfg["epidemic"]["observ_prob"]

# TODO: Gerar epidemia
infected_nodes = epidemic.si_epidemic(G, beta, init_infec, total_time, stop_frac)

observ_infec = epidemic.observed_infected(infected_nodes, observ_prob)


# 4. Compute network centralities
# TODO: Calcular métricas

degree = centrality.degree(G)
contact = centrality.contact(G, observ_infec)
betweenness = centrality.betweenness(G)
observ_betweenness = centrality.observed_betweenness(G, observ_infec)


# TODO: Salvar dados (usando a classe, e definir nome de arquivo)

