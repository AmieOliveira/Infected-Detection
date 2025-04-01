"""
    File containing the data formats for the project
"""

import os
import pickle
import torch
from torch.utils.data import Dataset


class EpidemicInstance:
    def __init__(self, graph, input_metrics, target, metadados):
        self.G = graph              # Lista de arestas
        self.X = input_metrics      # Valores dos inputs (infectados observáveis + métricas estruturais)
        # Ordenação dos valores em X: observáveis, grau, contact, betweenness, var-betweenness
        self.y = target             # Ground truth dos nós infectados
        self.metadados = metadados


class EpidemicDataset(Dataset):
    # TODO: classe para ler os .pkl de cada instãncia e concatenar num dat set só para a GNN

    # TODO: Docstring

    def __init__(self, folder, inputs):
        # TODO: Quais os argumentos que queremos dar de entrada? Precisamos inicializar o init da classe Dataset?
        #   Como fazer a organização dos dados?
        self.path = folder
        self.inputs = inputs
        self.data = []
        self.read_instances(folder)

    def read_instances(self, folder):
        # TODO: Read all .pkl files in the directory 'folder' and concatenate them accordingly

        if not os.path.exists(folder):
            raise ValueError(f"Directory '{folder}' not found")

        for filename in os.listdir(folder):
            if filename[-4:] != ".pkl":
                continue

            path = f"{folder}/{filename}"
            with open(path, "rb") as f:
                print(filename)
                ins = pickle.load(f)

                data_point = ins.G

                if not isinstance(ins, EpidemicInstance):
                    print(type(ins))
                    print(f"Error while reading file '{filename}'. May be corrupted?")

                metrics = [ins.X[inp].unsqueeze(-1) for inp in self.inputs]
                # (observed_tensor.unsqueeze(-1),
                # degree_tensor.unsqueeze(-1),
                # contact_tensor.unsqueeze(-1),
                # betweenness_tensor.unsqueeze(-1),
                # observ_betweenness_tensor.unsqueeze(-1))

                data_point.x = torch.cat(metrics, dim=-1)
                data_point.y = ins.y.unsqueeze(-1)

                self.data += [data_point]
                print(data_point)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.data[index]
