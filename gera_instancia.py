"""
    Script to generate an instance for the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np


parser = argparse.ArgumentParser(
    prog="Hidden network epidemics generator",
    description="Script designed to generate an instance of a network "
                "epidemie with hidden infected nodes. Customizations "
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

# 2. Create random graph
net_model = cfg["network"]["model"]

net = None
if net_model == "adjacency":
    net_path = cfg["network"]["adjacency"]["path"]
    # TODO: Ler grafo
elif net_model == "BA":
    m = cfg["network"]["BA"]["m"]
    # TODO: Criar grafo BA
elif net_model == "WS":
    k = cfg["network"]["WS"]["k"]
    p = cfg["network"]["WS"]["p"]
    # TODO: Criar grafo WS
elif net_model == "ER":
    p = cfg["network"]["ER"]["p"]
    # TODO: Criar grafo ER

# 3. Create epidemic
beta = cfg["epidemic"]["beta"]
observ_prob = cfg["epidemic"]["observ_prob"]
total_time = cfg["epidemic"]["total_time"]
stop_frac = cfg["epidemic"]["stop_frac"]

# TODO: Gerar epidemia

# TODO: Calcular métricas
# TODO: Salvar dados (usando a classe, e definir nome de arquivo)

