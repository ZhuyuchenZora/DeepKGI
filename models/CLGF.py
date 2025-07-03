import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_max_pool


class CLGF_GNNDrug(torch.nn.Module):
    def __init__(self, dim_drug):
        super().__init__()
        self.dim_drug = dim_drug

        # GCN layers (2 layers)
        self.gcn1 = GCNConv(77, dim_drug)
        self.gcn2 = GCNConv(dim_drug, dim_drug)
        self.relu = nn.ReLU()

        # GIN layers (3 layers)
        self.gin_layers = nn.ModuleList()
        self.gin_bns = nn.ModuleList()
        for i in range(3):
            mlp = nn.Sequential(
                nn.Linear(77 if i == 0 else dim_drug, dim_drug),
                nn.ReLU(),
                nn.Linear(dim_drug, dim_drug)
            )
            self.gin_layers.append(GINConv(mlp))
            self.gin_bns.append(nn.BatchNorm1d(dim_drug))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # === GCN Path ===
        x_gcn1 = self.relu(self.gcn1(x, edge_index))  # X_d^(1)
        x_gcn2 = self.relu(self.gcn2(x_gcn1, edge_index))  # X_d^(2)

        x_gcn_add = x_gcn1 + x_gcn2  # X_d^(1) + X_d^(2)
        x_gcn_mul = x_gcn1 * x_gcn2  # X_d^(1) ⨀ X_d^(2)

        j_mgcn = torch.cat([x_gcn1, x_gcn2, x_gcn_add, x_gcn_mul], dim=-1)  # shape: [N, 4*dim_drug]

        # === GIN Path ===
        h_list = []
        h = x
        for i in range(3):
            h = self.gin_layers[i](h, edge_index)
            h = self.relu(h)
            h = self.gin_bns[i](h)
            h_list.append(h)

        h1, h2, h3 = h_list
        h_add = h1 + h2 + h3
        h_mul = h1 * h2 * h3

        j_mgin = torch.cat([h1, h2, h3, h_add, h_mul], dim=-1)  # shape: [N, 5*dim_drug]

        # === Feature Fusion ===
        fused_node_repr = torch.cat([j_mgcn, j_mgin], dim=-1)  # shape: [N, (4+5)*dim_drug = 9*dim_drug]

        # === Graph-level Feature (Global Pooling) ===
        x_drug_graph = global_max_pool(fused_node_repr, batch)  # shape: [B, 9*dim_drug]

        return x_drug_graph

