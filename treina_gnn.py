#!/usr/bin/python

"""
    Script to train a GNN model for the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np
import torch
from formatacao import EpidemicDataset
from models import GCN, train


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
                "arguments given will overwrite the corresponding \n"
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
dat_parser.add_argument(
    "--batch-size", "--batch", type=int,
    help="Batch size parameter."
)

mod_parser = parser.add_argument_group(
    "GNN Model arguments",
    description="Arguments to configure the NN model to be used. Whatever \n"
                "arguments given will overwrite the corresponding \n"
                "configuration file inputs.",
)
mod_parser.add_argument(
    "--dim-layer", type=int,
    help="Dimension of the model hidden layer."
)
# TODO: Finish argparse

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

#   2.1 Main data set
dataset = EpidemicDataset(data_path, input_fields)

#   2.2 Split into train and test sets
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[0.8, 0.2],
    generator=torch.Generator().manual_seed(int(np.random.random()*100)),
)

#   2.3 Create train and test data loaders
batch_size = args.batch_size if args.batch_size else cfg["dataset"]["batch_size"]
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    pin_memory=True,
    num_workers=2,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    pin_memory=True,
    num_workers=2,
)

# 3. Create GNN model
input_dims = dataset[0].x.shape[-1]
dim_layer = args.dim_layer if args.dim_layer else cfg["model"]["dim_layer"]

lrate = 0.01  # TODO: Add parser and read from it and configuration file

model = GCN(
    dim_in=input_dims,
    dim_layer=dim_layer,
    dim_out=1
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

# 4. Train model
# FIXME: Bug no treinamento
train(
    model,
    train_loader,
    optimizer,
    device,
    1000
)
# FIXME: O treino não deveria receber o test_loader tb?

print("Reached here")

# TODO: Salvar modelo
# TODO: Avaliar modelo -- e salvar análises no conjunto de treino e de teste

