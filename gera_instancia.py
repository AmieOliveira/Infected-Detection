#!/usr/bin/python

"""
    Script to generate an instance for the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np
import networkx as nx
import pickle
import torch
from torch_geometric.utils import from_networkx

import epidemic
import centrality
from formatacao import EpidemicInstance


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
parser.add_argument(
    "--outpath", "--outdir", "-d",
    help="Directory where the output should be saved."
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
    "--beta", "-b", nargs="+", default=None,
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
    "--observ", "--observ-prob", "-o", nargs="+", default=None,
    help="Probability of an infected node being identified as \n"
         "infected"
)

# 1. Read from configuration file
args = parser.parse_args()
config_file = args.configfile
print(f"Reading configurations from {config_file}")
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

seed = cfg["rd_seed"]
if args.seed:
    seed += args.seed
np.random.seed(seed)

metadados = {
    "Random seed": seed,
}

# 2. Create graph
net_model = args.model_type
if not net_model:
    net_model = cfg["network"]["model"]

outpath = args.outpath
if not outpath:
    outpath = f"results/{net_model}"
# TODO: Add output path to config file

metadados["model"] = {
        "type": net_model
}

G = None
if net_model == "adjacency":  # TODO: Not fully implemented
    raise ValueError(f"Generating an instance with a predefined adjacency matrix is not correctly implemented yet!")

    net_path = args.path
    if not net_path:
        net_path = cfg["network"]["adjacency"]["path"]
        adj_array = None  # FIXME: Read from net_path
    G = nx.from_numpy_matrix(np.array(adj_array))
    metadados["model"]["path"] = net_path
else:
    n = args.n_nodes
    if not n:
        n = cfg["network"]["n_nodes"]
    metadados["model"]["n"] = n

    if net_model == "BA":
        m = args.BA_m
        if not m:
            m = cfg["network"]["BA"]["m"]
        G = nx.barabasi_albert_graph(n, m)
        metadados["model"]["m"] = m

    elif net_model == "WS":
        k = args.WS_k
        if not k:
            k = cfg["network"]["WS"]["k"]
        p = args.WS_p
        if not p:
            p = cfg["network"]["WS"]["p"]
        G = nx.watts_strogatz_graph(n, k, p)
        metadados["model"]["p"] = p
        metadados["model"]["k"] = k

    elif net_model == "ER":
        p = args.WS_p
        if not p:
            p = cfg["network"]["ER"]["p"]
        G = nx.erdos_renyi_graph(n, p)
        metadados["model"]["p"] = p

    else:
        raise ValueError(f"Model {net_model} not implemented.")

print(f"Created graph: {G}")

# 3. Create epidemic
beta_choices = args.beta if args.beta else cfg["epidemic"]["beta"]
observ_prob_choices = args.observ if args.observ else cfg["epidemic"]["observ_prob"]

beta = np.random.choice(beta_choices, 1)[0]
observ_prob = np.random.choice(observ_prob_choices, 1)[0]
print(f"Infection rate is {beta}, observation probability is {observ_prob}")

init_infec = args.initial_infection if args.initial_infection else cfg["epidemic"]["init_infec"]

total_time = args.time if args.time else cfg["epidemic"]["total_time"]
tt = total_time if total_time > 0 else None

stop_frac = args.stop if args.stop else cfg["epidemic"]["stop_frac"]
sf = stop_frac if stop_frac > 0 else None

metadados["epidemic"] = {
    "beta": beta,
    "init_infec": init_infec,
    "total_time": total_time,
    "stop_frac": stop_frac,
    "observ_prob": observ_prob,
    "beta_choices": beta_choices,
    "observ_prob_choices": observ_prob_choices,
}

infected_nodes = epidemic.si_epidemic(G, beta, init_infec, tt, sf)

observ_infec = epidemic.observed_infected(infected_nodes, observ_prob)
print(f"Generated epidemic! {len(infected_nodes)} infected and {len(observ_infec)} observed.")


# 4. Compute network centralities
degree = centrality.degree(G)
contact = centrality.contact(G, observ_infec)
contact_k_vector = centrality.contact_k_vector(G, observ_infec, 3)
contact_k = centrality.contact_k(G, observ_infec, 2)
betweenness = centrality.betweenness(G)
observ_betweenness = centrality.observed_betweenness(G, observ_infec)
print("Finished calculating the centrality metrics")

# 5. Create tensor
observed_tensor = torch.full((len(G),), 0.0)
for node in observ_infec:
    observed_tensor[node] = 1

degree_tensor = torch.tensor(list(degree.values()))
contact_tensor = torch.tensor(list(contact.values()))
contact_k_vector_2_tensor = torch.tensor([fractions[1] for fractions in contact_k_vector.values()])
contact_k_vector_3_tensor = torch.tensor([fractions[2] for fractions in contact_k_vector.values()])
contact_k_tensor = torch.tensor(list(contact_k.values()))
betweenness_tensor = torch.tensor(list(betweenness.values()))
observ_betweenness_tensor = torch.tensor(list(observ_betweenness.values()))

input_metrics = {
    "OBS_I": observed_tensor,
    "DEG": degree_tensor,
    "CONT": contact_tensor,
    "CONT_2": contact_k_vector_2_tensor,
    "CONT_3": contact_k_vector_3_tensor,
    "CONT_k2": contact_k_tensor,
    "BETW": betweenness_tensor,
    "OBS_B": observ_betweenness_tensor,
}

Y = torch.full((len(G),), 0.0)
Y[infected_nodes] = 1
print("Generated GNN data")
print(f"Metadata: {metadados}")

# 6. Format and save data
params = ""
keys = list(metadados["model"].keys())
keys.sort()
for key in keys:
    if key != "type":
        params += f"{key}{metadados['model'][key]}"
epinfo = f"ii{init_infec}-t{total_time}-f{stop_frac}"
# Note: Removed beta and observ_prob values from instance name due to the cluster setup

runIdx = f"run{args.seed}" if (args.seed is not None) else f"s{seed}"
outfile = f"{outpath}/instance_model{net_model}-{params}_epidemic-{epinfo}_{runIdx}.pkl"

graph_data = from_networkx(G)

with open(outfile, 'wb') as of:
    datum = EpidemicInstance(graph_data, input_metrics, Y, metadados)
    pickle.dump(datum, of, pickle.HIGHEST_PROTOCOL)

print(f"Saved output to {outfile}")

