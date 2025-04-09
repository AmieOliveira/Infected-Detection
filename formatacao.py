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


import os
import pickle
import torch
from torch.utils.data import Dataset  # ou de torch_geometric.data, se necessário
#from epidemic_instance import EpidemicInstance  # ajuste conforme onde estiver definido

class EpidemicDataset(Dataset):
    """
    Dataset para carregar arquivos .pkl contendo instâncias da simulação epidêmica,
    gerando grafos com features para treinar GNNs.
    """

    def __init__(self, folder, inputs):
        """
        Args:
            folder (str): Caminho da pasta contendo os arquivos .pkl.
            inputs (list[str]): Lista com os nomes das features a serem usadas como entrada (ex: ['OBS_I', 'DEG', ...])
        """
        self.path = folder
        self.inputs = inputs
        self.data = []
        self.read_instances(folder)

    def read_instances(self, folder):
        """
        Lê todos os arquivos .pkl no diretório e extrai os grafos (data_point) com
        features combinadas.
        """
        if not os.path.exists(folder):
            raise ValueError(f"Directory '{folder}' not found")

        for filename in os.listdir(folder):
            if not filename.endswith(".pkl"):
                continue

            path = os.path.join(folder, filename)
            with open(path, "rb") as f:
                ins = pickle.load(f)

            if not isinstance(ins, EpidemicInstance):
                print(f"⚠️ Warning: File '{filename}' is not a valid EpidemicInstance (got {type(ins)}). Skipping.")
                continue

            data_point = ins.G

            # Concatena os tensores de entrada
            try:
                metrics = [ins.X[inp].unsqueeze(-1) for inp in self.inputs]
                data_point.x = torch.cat(metrics, dim=-1)
                data_point.y = ins.y.float().unsqueeze(-1)  # Saída esperada: [N, 1]
                self.data.append(data_point)
            except KeyError as e:
                print(f"❌ KeyError: '{e}' não encontrado em ins.X ao processar '{filename}'. Pulando arquivo.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def all_data(self):
        return self.data

