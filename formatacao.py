"""
    File containing the data formats for the project
"""

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

    def __init__(self):
        # TODO: Quais os argumentos que queremos dar de entrada? Precisamos inicializar o init da classe Dataset?
        #   Como fazer a organização dos dados?
        ...

    def read_instances(self, folder):
        # TODO: Read all .pkl files in the directory 'folder' and concatenate them accordingly
        ...

    def __getitem__(self, index):
        # TODO: Method to return one instance of the data set
        ...
