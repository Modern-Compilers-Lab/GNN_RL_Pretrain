"""
THE FOLLOWING NEEDS TO BE UPDATED. THE INFO BELOW MAY BE OUTDATED.
Graph Convolutional Network (GCN) implementation for Agent in PEARL.

OPTIMAL CONFIGURATION:
{
    'input_size': 175,
    'hidden_size': 90,
    'num_gcn_layers': 3,
    'num_outputs': 32,
    'dropout_prob': 0.1,
    'use_batch_norm': True
}

MODEL STATS:
- Parameters: 206,763 (4.8% smaller than GAT)
- Size: 0.79 MB
- Architecture: 3 GCN layers + policy/value heads
"""

from torch_geometric.nn import (
    GCNConv,
    Linear,
)

from neural_nets.pooling import GlobalPooling, SAGPoolLayer, DiffPoolLayer

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.distributions import Categorical


class GCN(nn.Module):
    """
    Graph Convolutional Network for Tiramisu program optimization.
    
    Uses graph convolutions to process program dependency graphs
    for predicting execution times and optimizing compiler schedules.
    
    Designed for small graphs with rich node features (175-dimensional).
    """
    
    def __init__(
        self,
        input_size=175,      # Node feature dimension from dataset
        hidden_size=90,      # Hidden layer size (optimal: 90)
        num_outputs=32,      # Action space size
        dropout_prob=0.1,    # Dropout probability
        use_batch_norm=True, # Whether to use batch normalization
        num_gcn_layers=3,    # Number of GCN layers (optimal: 3)
        pooling_type='global', # Type of pooling layer
        num_clusters=10,     # Number of clusters for DiffPool
        sag_ratio=0.5,       # Ratio for SAGPool
    ):
        super(GCN, self).__init__()
        
        self.dropout = nn.AlphaDropout(dropout_prob)
        self.use_batch_norm = use_batch_norm
        self.num_gcn_layers = num_gcn_layers
        self.pooling_type = pooling_type
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.linear_projections = nn.ModuleList()
        
        # First GCN layer
        self.gcn_layers.append(GCNConv(input_size, hidden_size))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_size))
        self.linear_projections.append(Linear(hidden_size, hidden_size))
        
        # Additional GCN layers
        for i in range(1, num_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_size, hidden_size))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_size))
            self.linear_projections.append(Linear(hidden_size, hidden_size))

        # Pooling layers
        self.pooling_layers = nn.ModuleList()
        total_graph_features = 0
        for _ in range(num_gcn_layers):
            if pooling_type == 'global':
                pool_layer = GlobalPooling(hidden_size)
            elif pooling_type == 'sag':
                pool_layer = SAGPoolLayer(hidden_size, ratio=sag_ratio)
            elif pooling_type == 'diff':
                pool_layer = DiffPoolLayer(hidden_size, num_clusters)
            else:
                raise ValueError(f"Unsupported pooling type: {pooling_type}")
            self.pooling_layers.append(pool_layer)
            total_graph_features += pool_layer.get_output_dim()
        
        # Initialize GCN weights
        for gcn_layer in self.gcn_layers:
            for name, param in gcn_layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
        
        # Initialize linear projection weights
        for linear_layer in self.linear_projections:
            nn.init.xavier_uniform_(linear_layer.weight)
        
        # Graph-level representation combiner
        self.graph_combiner = Linear(total_graph_features, hidden_size * 2)
        nn.init.xavier_uniform_(self.graph_combiner.weight)
        
        # Shared representation layer
        self.shared_linear = Linear(hidden_size * 2, hidden_size)
        nn.init.xavier_uniform_(self.shared_linear.weight)
        
        # Policy network (action probabilities)
        self.π = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, num_outputs), std=0.1),
        )
        
        # Value network (execution time estimation)
        self.v = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, 1)),
        )
    
    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize layer weights with orthogonal initialization"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def shared_layers(self, data):
        """
        Extract shared graph-level representation from input data.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [num_nodes, input_size]
                - edge_index: edge connectivity [2, num_edges]
                - batch: batch assignment [num_nodes]
        
        Returns:
            graph_representation: [batch_size, hidden_size]
        """
        x, edge_index, batch_index = data.x, data.edge_index, data.batch
        
        layer_representations = []
        total_link_loss = 0
        total_ent_loss = 0
        
        # Process through each GCN layer
        for i in range(self.num_gcn_layers):
            # GCN convolution
            x = self.gcn_layers[i](x, edge_index)
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Activation and linear projection
            x = nn.functional.selu(x)
            x = self.linear_projections[i](x)
            x = nn.functional.selu(x)
            
            # Apply pooling
            graph_repr, link_loss, ent_loss = self.pooling_layers[i](x, edge_index, batch_index)
            layer_representations.append(graph_repr)

            if link_loss is not None:
                total_link_loss += link_loss
            if ent_loss is not None:
                total_ent_loss += ent_loss
        
        # Combine representations from all layers
        combined_repr = torch.cat(layer_representations, dim=-1)
        
        # Final transformation
        graph_features = self.graph_combiner(combined_repr)
        graph_features = nn.functional.selu(self.shared_linear(graph_features))
        graph_features = self.dropout(graph_features)
        
        return graph_features, total_link_loss, total_ent_loss
    
    def forward(self, data, actions_mask=None, action=None):
        """
        Forward pass for policy and value estimation.
        
        Args:
            data: PyTorch Geometric Data object
            actions_mask: Optional mask for invalid actions
            action: Optional specific action (for evaluation)
        
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: Estimated value (execution time)
        """
        # Extract shared graph representation
        weights, link_loss, ent_loss = self.shared_layers(data)
        
        # Policy computation
        logits = self.π(weights)
        
        # Apply action mask if provided
        if actions_mask is not None:
            logits = logits - actions_mask * 1e8
        
        # Create categorical distribution
        probs = Categorical(logits=logits)
        
        # Sample action if not provided
        if action is None:
            action = probs.sample()
        
        # Value estimation
        value = self.v(weights)
        
        # Add pooling losses if applicable
        loss = 0
        if self.pooling_type == 'diff':
            loss += link_loss + ent_loss

        return action, probs.log_prob(action), probs.entropy(), value, loss
