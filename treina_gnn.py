#!/usr/bin/python

"""
    Script to train a GNN model for the infected nodes detection problem
"""

import argparse
import yaml
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from formatacao import EpidemicDataset
from models import GCN, train
from avaliacao import auc_statistics

print("Starting script")

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
    "--batch-size", "--batch", "-b", type=int,
    help="Batch size parameter."
)
dat_parser.add_argument(
    "--split-seed", "--sSeed", type=int,
    help="Random seed for the data set splitting."
)
dat_parser.add_argument(
    "--split-prop", "--sProp", type=int,
    help="Proportion of the data set that should be used for training."
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
mod_parser.add_argument(
    "--learning-rate", "--lr", type=float,
    help="Training learning rate"
)
mod_parser.add_argument(
    "--epochs", "-e", type=float,
    help="Number of training epochs to be performed."
)
# TODO: Finish argparse

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Read from configuration file
config_file = args.configfile
print(f"Reading configurations from {config_file}")
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

seed = args.seed
if not seed:
    seed = cfg["rd_seed"]
np.random.seed(seed)


# 2. Create data set
data_path = args.indir if args.indir else cfg["dataset"]["path"]
input_fields = args.input_fields if args.input_fields else cfg["dataset"]["inputs"]
print(f"Getting data from folder {data_path}.")
print(f"Model will be trained with the following inputs: {input_fields}")
# FIXME: Quero colocar hardcoded que o OBS_I deve ser um dos inputs?

#   2.1 Main data set
dataset = EpidemicDataset(data_path, input_fields)

print(f"Created data set of length {len(dataset)}")

#   2.2 Split into train and test sets
split_seed = args.split_seed if args.split_seed else cfg["dataset"]["split_seed"]
split_prop = args.split_prop if args.split_prop else cfg["dataset"]["split_prop"]
if split_prop > 1 or split_prop < 0:
    raise ValueError(f"Split proportion should be a number between 0 and 1 and {split_prop} was provided")

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset=dataset,
    lengths=[split_prop, 1 - split_prop],
    generator=torch.Generator().manual_seed(split_seed),
)
print(f"Split data into train ({len(train_dataset)} instances) and test ({len(test_dataset)} instances) sets")

#   2.3 Create train and test data loaders
batch_size = args.batch_size if args.batch_size else cfg["dataset"]["batch_size"]
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
)

# 3. Create GNN model
input_dims = dataset[0].x.shape[-1]
dim_layer = args.dim_layer if args.dim_layer else cfg["model"]["dim_layer"]
l_rate = args.learning_rate if args.learning_rate else cfg["model"]["training"]["learning_rate"]

model = GCN(
    dim_in=input_dims,
    dim_layer=dim_layer,
    dim_out=1
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

# 4. Train model
n_epochs = args.epochs if args.epochs else cfg["model"]["training"]["n_epochs"]

train(
    model,
    train_loader,
    optimizer,
    device,
    n_epochs,
)
# FIXME: Loss só aumenta deve ter algum bug

# FIXME: O treino não deveria receber o test_loader tb?

print("Successfully finished training!")

# TODO: Salvar modelo

# TODO: Avaliar modelo
stats_train = auc_statistics(train_dataset, model)
if len(test_dataset) > 0:
    stats_test = auc_statistics(test_dataset, model)

# TODO: Terminar as funções de avaliação e salvar os resultados (definir nome dos arquivos, etc.)
