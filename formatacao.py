"""
    File containing the data formats for the project
"""

import os
import pickle
import torch
from torch_geometric.data import Dataset


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

    # FIXME: Será que seria melhor ler o arquivo na hora que fosse usar só? (Por questões de memória) Isso seria factível?

    def __init__(self, folder, inputs):
        # TODO: Quais os argumentos que queremos dar de entrada? Precisamos inicializar o init da classe Dataset?
        #   Como fazer a organização dos dados?
        self.path = folder
        self.inputs = inputs
        self.data = []
        # self.transform = transforms.Compose([transforms.ToTensor()])
        self.read_instances(folder)

    def read_instances(self, folder):
        # TODO: Read all .pkl files in the directory 'folder' and concatenate them accordingly

        if not os.path.exists(folder):
            raise ValueError(f"Directory '{folder}' not found")

        for filename in os.listdir(folder):
            # TODO: Add logger, and this message in the debug setting
            # print(f"\t EpidemicDataset: Reading from file {filename}")

            if filename[-4:] != ".pkl":
                # print("\t EpidemicDataset: \tDisregarding file")
                continue

            path = f"{folder}/{filename}"
            with open(path, "rb") as f:
                # print(filename)
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
                # print(data_point)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def all_data(self):
        return self.data
