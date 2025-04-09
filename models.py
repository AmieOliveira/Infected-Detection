"""
    File containing the AI models for the project
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import BCEWithLogitsLoss


# TODO: Conferir a classe
class GCN(torch.nn.Module):
    # TODO: Documentação
    def __init__(
            self, dim_in=128, dim_layer=128, dim_out=1
    ):
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_layer)
        self.conv2 = GCNConv(dim_layer, dim_out)    # ultima dimensao vai ser a dimensao de G.H

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.sigmoid(x)
        return x


def train(model, loader, optimizer, device,epochs):    
    criterion = BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            y = data.y.float().view(-1, 1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader)}')
