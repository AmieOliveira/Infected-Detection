#!/usr/bin/python

"""
    Script to train a GNN model for the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np
import torch
from formatacao import EpidemicDataset


parser = argparse.ArgumentParser(
    prog="Pipeline for GNN training",
    description="Script designed to train GNNs to detect hidden \n"
                "infected nodes in a network epidemic. Customizations \n"
                "should be given via a configuration file and \n"
                "command line prompt (optional)."
)

parser.add_argument(
    "configfile",
    default="treina_gnn_config.yaml", nargs="?",
    help="Name of the configuration file to be used. If none is \n"
         "given, then the default 'treina_gnn_config.yaml' will \n"
         "be used."
)
parser.add_argument(
    "--seed", "-s", type=int,
    help="Random seed to be used in the data generation."
)
parser.add_argument(
    "--outpath", "--outdir", "-d",
    help="Directory where the output should be saved."
)
# TODO: Finish argparse

args = parser.parse_args()

# 1. Read from configuration file
# TODO: Configuration file
config_file = args.configfile
print(f"Reading configurations from {config_file}")
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

seed = args.seed
if not seed:
    seed = cfg["rd_seed"]
np.random.seed(seed)  # TODO: Conferir se isso é o suficiente

# TODO: Criar dataset (load dos dados, separação, data loader)
# TODO: Criar modelo
# TODO: Treinar modelo
# TODO: Salvar modelo
# TODO: Avaliar modelo??

