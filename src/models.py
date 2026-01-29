import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, global_mean_pool

class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )))

        # Output layer
        self.layers.append(GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATv2Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads=1, dropout=0.5):
        super(GATv2Encoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Output layer
        self.layers.append(GATv2Conv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HybridGIN_GAT_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads=1, dropout=0.5):
        super(HybridGIN_GAT_Model, self).__init__()
        self.gin_encoder = GINEncoder(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.gat_encoder = GATv2Encoder(input_dim, hidden_dim, output_dim, num_layers, heads, dropout)

        self.graph_rep = nn.Linear(output_dim * 2, output_dim)
        self.classifier = nn.Linear(output_dim + 1, 1)  # Binary classification + Lovász

        self.ortho_reg = nn.MSELoss()
        self.handle_reg = nn.MSELoss()

    def forward(self, x, edge_index, batch):
        gin_node_rep = self.gin_encoder(x, edge_index)
        gat_node_rep = self.gat_encoder(x, edge_index)

        node_rep = torch.cat([gin_node_rep, gat_node_rep], dim=1)
        graph_rep = global_mean_pool(node_rep, batch)
        graph_rep = self.graph_rep(graph_rep)

        subgraph_lovasz = self.compute_subgraph_lovasz(node_rep, graph_rep, edge_index, batch)
        graph_rep = torch.cat([graph_rep, subgraph_lovasz.unsqueeze(1)], dim=1)

        logits = self.classifier(graph_rep)
        return logits, graph_rep

    # Placeholder Lovász methods
    def compute_subgraph_lovasz(self, node_rep, graph_rep, edge_index, batch):
        return torch.zeros(batch.max().item() + 1, device=node_rep.device)
    
    def compute_lovasz_loss(self, node_rep, graph_rep, edge_index, batch):
        ortho_loss = self.ortho_reg(node_rep @ node_rep.T, torch.eye(node_rep.size(0)))
        handle_loss = self.handle_reg(torch.norm(graph_rep, dim=1), torch.ones(graph_rep.size(0)))
        return ortho_loss + handle_loss
