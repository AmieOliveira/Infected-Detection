"""
    File containing the AI models for the project
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BCEWithLogitsLoss
import copy
import pandas as pd


class GCN(torch.nn.Module):
    # TODO: Documentação
    def __init__(
            self, dim_in=128, dim_layer=128, dim_out=1
    ):
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_layer)
        self.conv2 = GCNConv(dim_layer, dim_out)
        #self.conv3 = GCNConv(dim_layer, dim_out)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)        
        x = F.sigmoid(x)
        return x


def val(model, val_loader, device):
    """
    Avaliação do modelo no conjunto de validação.
    Retorna a loss média.
    """
    model.eval()
    criterion = BCEWithLogitsLoss()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            y = data.y.float().view(-1, 1)
            x = data.x[:, 0].unsqueeze(-1)
            mask = x != y
            loss = criterion(out[mask], y[mask])
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(model, train_loader, val_loader, optimizer, device, epochs):
    criterion = BCEWithLogitsLoss()
    best_model = None
    best_val = None
    val_losses = []

    # DataFrame para armazenar histórico
    log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            y = data.y.float().view(-1, 1)
            x = data.x[:, 0].unsqueeze(-1)
            mask = x != y
            if mask.sum() == 0:
                continue
            loss = criterion(out[mask], y[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = None

        # Avaliação de validação a cada 50 épocas
        if (epoch + 1) % 50 == 0:
            val_loss = val(model, val_loader, device)
            val_losses.append(val_loss)

            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_model = copy.deepcopy(model)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # Salva no DataFrame
        log_df.loc[len(log_df)] = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        }

    return best_model, log_df

