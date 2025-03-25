"""
    File containing the data formats for the project
"""

import torch


class EpidemicDataset:
    def __init__(self, graph, input_metrics, target, metadados):
        self.G = graph              # Lista de arestas
        self.X = input_metrics      # Valores dos inputs (infectados observáveis + métricas estruturais)
        # Ordenação dos valores em X: observáveis, grau, contact, betweenness, var-betweenness
        self.y = target             # Ground truth dos nós infectados
        self.metadados = metadados


# TODO: Criar script de gerar uma instância e salvar os dados no formato EpidemicDataset

# TODO: Criar objeto Dataset (baseado no torch) que concatena os vários arquivos

