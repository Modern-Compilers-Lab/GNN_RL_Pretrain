import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    SAGPooling,
    GCNConv,
)
from torch_geometric.utils import to_dense_adj, to_dense_batch

class GlobalPooling(nn.Module): # working fine
    """
    Combines global mean, max, and sum pooling.
    """
    def __init__(self, in_channels):
        super(GlobalPooling, self).__init__()
        self.in_channels = in_channels

    def forward(self, x, edge_index, batch):
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_sum = global_add_pool(x, batch)
        
        # Concatenate the pooled features
        graph_repr = torch.cat([graph_mean, graph_max, graph_sum], dim=-1)
        
        return graph_repr, None, None

    def get_output_dim(self):
        # Mean, max, and sum pooling are concatenated
        return self.in_channels * 3

class SAGPoolLayer(nn.Module): # working fine
    """
    Self-Attention Pooling (SAGPool).
    """
    def __init__(self, in_channels, ratio=0.5):
        super(SAGPoolLayer, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.in_channels = in_channels

    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.sag_pool(x, edge_index, batch=batch)
        
        # After pooling, apply global pooling to get a graph-level representation
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_sum = global_add_pool(x, batch)
        
        graph_repr = torch.cat([graph_mean, graph_max, graph_sum], dim=-1)
        return graph_repr, None, None

    def get_output_dim(self):
        return self.in_channels * 3

class DiffPoolLayer(nn.Module): # gives error, needs debugging
    """
    Differentiable Pooling (DiffPool).
    
    This implementation uses a GCN to learn cluster assignments.
    """
    def __init__(self, in_channels, num_clusters):
        super(DiffPoolLayer, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        
        # GNN for cluster assignment
        self.assignment_gnn = GCNConv(in_channels, num_clusters)

    def forward(self, x, edge_index, batch):
        # Get cluster assignments
        s = self.assignment_gnn(x, edge_index).softmax(dim=-1)
        
        # Convert to dense representations for DiffPool
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        s_dense, _ = to_dense_batch(s, batch)
        
        # Differentiable pooling
        x_pooled, adj_pooled, link_loss, ent_loss = self.dense_diff_pool(x_dense, adj_dense, s_dense, mask)
        
        # Flatten the pooled features to get a graph-level representation
        graph_repr = x_pooled.view(x_pooled.size(0), -1)
        
        return graph_repr, link_loss, ent_loss

    def get_output_dim(self):
        return self.in_channels * self.num_clusters

    @staticmethod
    def dense_diff_pool(x, adj, s, mask=None):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()

        s = torch.softmax(s, dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x = x * mask
            s = s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / adj.numel()

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

        return out, out_adj, link_loss, ent_loss
