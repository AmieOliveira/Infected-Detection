#!/usr/bin/python

"""
    Script to evaluate a trained GNN model on a specific data set for
    the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np
import torch
import random
import os
from torch_geometric.loader import DataLoader
from formatacao import EpidemicDataset
from avaliacao import auc_statistics

parser = argparse.ArgumentParser(
    prog="Pipeline for GNN evaluation",
    description="Script designed to evaluate trained GNNs to detect hidden \n"
                "infected nodes in a network epidemic. Customizations \n"
                "should be given via a configuration file and \n"
                "command line prompt (optional)."
)

parser.add_argument(
    "configfile",
    default="avalia_modelo_config.yaml", nargs="?",
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

# TODO: Argparser
# TODO: Add logger

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Read from configuration file
config_file = args.configfile
print(f"Reading configurations from '{config_file}'")
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

seed = cfg["rd_seed"]
if args.seed:
    seed += args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

outpath = args.outpath
if not args.outpath:
    outpath = cfg["output_path"]

metadados = {
    "Random seed": seed,
}

# 2. Read from model
# TODO: Add argparser options
model_name = cfg["model"]["name"]
model_path = cfg["model"]["path"]
model_base = cfg["model"]["filebase"]

model_file = os.path.join(model_path, f"model_{model_base}.gnn")
print(f"Getting model from {model_file}")
model = torch.load(model_file, weights_only=False)

print(f"Extracted model: {model}. Name: {model_name}")

# 3. Read model info
model_infofile = os.path.join(model_path, f"stats_{model_base}.dict")
with open(model_infofile, "r") as f:
    model_info = yaml.load(f, Loader=yaml.SafeLoader)["config"]

print(f"Extracted model info: {model_info}")

# 3. Create data set object
# TODO: Add argparser options
data_name = cfg["dataset"]["name"]
data_path = cfg["dataset"]["path"]
input_fields = model_info["dataset"]["inputs"]  # TODO: Get from model info

print(f"Getting data from folder {data_path}.")
dataset = EpidemicDataset(data_path, input_fields)
print(f"Created data set with {len(dataset)} instances")

# 4. Evaluate model
input_fields = dataset.inputs
auc = auc_statistics(dataset, model, input_fields, device)
print(f"Average AUC: {auc['GNN']['mean']}")

stats = {
    "validation": auc['GNN'],
    "config": model_info,
    "model": {"name": model_name, "path": model_file},
    "dataset": {"name": data_name, "path": data_path},
}

for key in auc.keys():
    if key == "GNN":
        continue
    stats[key] = auc[key]

print("Statistics calculated: ")
print(stats)
print()

# 5. Save statistics
filename = f"stats_model-{model_name}_data-{data_name}.dict"
# TODO: Rever output name

outfile = os.path.join(outpath, filename)

with open(outfile, "w") as yamlfile:
    data = yaml.dump(stats, yamlfile)
    print(f"Wrote evaluation statistics to file: {outfile}")

