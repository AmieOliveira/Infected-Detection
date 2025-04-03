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
    "--outpath", "--outdir", "-o",
    help="Directory where the output should be saved."
)

dat_parser = parser.add_argument_group(
    "Dataset arguments",
    description="Arguments to select the data set to be used. Whatever \n"
                "arguments gicen will overwrite the corresponding \n"
                "configuration file inputs.",
)
dat_parser.add_argument(
    "--indir", "--datadir", "-i",
    help="Directory where the input data is stored."
)
dat_parser.add_argument(
    "--input-fields", "--inputs", nargs="+",
    choices=["OBS_I", "DEG", "CONT", "BETW", "OBS_B"],
    help="List all input variables to be given to the model \n"
         "(separated by spaces. Note that the observed infected \n"
         "nodes ('OBS_I') should be always provided as input.",
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


# 2. Create data set
data_path = args.indir if args.indir else cfg["dataset"]["path"]
input_fields = args.input_fields if args.input_fields else cfg["dataset"]["inputs"]
print(f"Getting data from folder {data_path}.")
print(f"Model will be trained with the following inputs: {input_fields}")
# FIXME: Quero colocar hardcoded que o OBS_I deve ser um dos inputs?

dataset = EpidemicDataset(data_path, input_fields)
# TODO: Separação em treino e teste, criação dos data loaders

# TODO: Criar modelo
# TODO: Treinar modelo
# TODO: Salvar modelo
# TODO: Avaliar modelo??

