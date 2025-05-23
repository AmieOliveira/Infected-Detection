"""
    File containing the data formats for the project
"""

import os
import pickle
import torch
from torch_geometric.data import Dataset


# TODO: Add logger

class EpidemicInstance:
    def __init__(self, graph, input_metrics, target, metadados):
        self.G = graph                      # Lista de arestas
        self.X = input_metrics              # Tensor [n_nodes, n_features]
        self.y = target                     # Ground truth dos infectados
        self.metadados = metadados


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
        self.data = []
        self.inputs = inputs
        if "OBS_I" not in inputs:
            self.inputs = ["OBS_I"] + inputs
            print("Warning: observed infected not in the inputs. They will be inserted by default. "
                  f"New input list: {self.inputs}")
        elif inputs[0] != "OBS_I":
            self.inputs.remove("OBS_I")
            self.inputs = ["OBS_I"] + inputs
            print(f"Debug: Reordered input list: {self.inputs}")

        self.obs_b_pos = None if not ("OBS_B" in self.inputs) else self.inputs.index("OBS_B")
        self.cont_pos = None if not ("CONT" in self.inputs) else self.inputs.index("CONT")
        self.cont_k2_pos = None if not ("CONT_k2" in self.inputs) else self.inputs.index("CONT_k2")

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

                X_not_normalized = torch.cat(metrics, dim=-1)  # Tensor [n_nodes, n_features]
                # Copia a coluna 0 sem normalizar
                X0 = X_not_normalized[:, 0:1]  # Infectados observáveis (mantido)
                X_rest = X_not_normalized[:, 1:]  # Métricas estruturais (normalizar)
                # Normalização z-score nas demais colunas
                mean = X_rest.mean(dim=0)
                std = X_rest.std(dim=0)
                std[std == 0] = 1.0  # Evita divisão por zero
        
                X_rest_normalized = (X_rest - mean) / std
                # Concatena: coluna 0 original + demais normalizadas
                X = torch.cat([X0, X_rest_normalized], dim=1)
                
                data_point.x = X#torch.cat(metrics, dim=-1)
                
                data_point.y = ins.y.float().unsqueeze(-1)  # Saída esperada: [N, 1]
                
                if not self.obs_b_pos:
                    data_point.obs_b = ins.X["OBS_B"].unsqueeze(-1)
                if not self.cont_pos:
                    data_point.cont = ins.X["CONT"].unsqueeze(-1)
                if not self.cont_k2_pos:
                    data_point.cont_k2 = ins.X["CONT_k2"].unsqueeze(-1)

                self.data.append(data_point)

            except KeyError as e:
                print(f"❌ KeyError: '{e}' não encontrado em ins.X ao processar '{filename}'. Pulando arquivo.")

    def len(self):
        return len(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def all_data(self):
        return self.data
    
    def get(self, idx):
        return self.data[idx]

    def get_observed_betweenness(self, idx):
        if self.obs_b_pos:
            return self.data[idx].x[self.obs_b_pos]
        else:
            return self.data[idx].obs_b

    def get_contact(self, idx):
        if self.cont_pos:
            return self.data[idx].x[self.cont_pos]
        else:
            return self.data[idx].cont

    def get_2neighborhood_contact(self, idx):
        if self.cont_k2_pos:
            return self.data[idx].x[self.cont_k2_pos]
        else:
            return self.data[idx].cont_k2
