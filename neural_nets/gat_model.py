"""
Graph Attention Network (GAT) implementation for PEARL.

OPTIMAL CONFIGURATION:
{
    'input_size': 175,
    'hidden_size': 64, 
    'num_heads': 4,
    'num_outputs': 32,
    'dropout_prob': 0.1
}

MODEL STATS:
- Parameters: 217,249
- Size: 0.83 MB
- Architecture: 2 GAT layers + policy/value heads
"""

from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    Linear,
    global_max_pool,
)

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class GAT(nn.Module):
    """
    Graph Attention Network for Tiramisu program optimization.
    
    Uses attention mechanisms to focus on important program dependencies
    for predicting execution times and optimizing compiler schedules.
    """
    
    def __init__(
        self,
        input_size=175,      # Node feature dimension from dataset
        hidden_size=64,      # Hidden layer size
        num_heads=4,         # Number of attention heads
        num_outputs=32,      # Action space size
        dropout_prob=0.1,    # Dropout probability
    ):
        super(GAT, self).__init__()

        self.dropout = nn.AlphaDropout(dropout_prob)

        # First GAT layer
        self.conv_layer1 = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear1 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )
        
        # Second GAT layer
        self.conv_layer2 = GATv2Conv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear2 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )

        # Initialize GAT weights
        convolutions_layers = [self.conv_layer1, self.conv_layer2]
        for convlayer in convolutions_layers:
            for name, param in convlayer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)

        # Initialize linear projection weights
        linear_layers = [self.linear1, self.linear2]
        for linearlayer in linear_layers:
            nn.init.xavier_uniform_(linearlayer.weight)

        # Graph-level representation combiner
        self.convs_summarizer = Linear(
            in_channels=4 * hidden_size,  # 2 layers * 2 pooling ops * hidden_size
            out_channels=hidden_size * 2,
        )

        # Shared representation layer
        self.shared_linear1 = Linear(
            in_channels=hidden_size * 2, 
            out_channels=hidden_size
        )
        nn.init.xavier_uniform_(self.shared_linear1.weight)

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
        Extract shared graph-level representation using attention.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            graph_representation: [batch_size, hidden_size]
        """
        x, edges_index, batch_index = (data.x, data.edge_index, data.batch)

        # First GAT layer
        x = self.conv_layer1(x, edges_index)
        x = nn.functional.selu(self.linear1(x))
        x1 = torch.cat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        # Second GAT layer
        x = self.conv_layer2(x, edges_index)
        x = nn.functional.selu(self.linear2(x))
        x2 = torch.cat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        # Combine layer representations
        x = torch.cat((x1, x2), dim=-1)

        # Final transformation
        x = self.convs_summarizer(x)
        x = nn.functional.selu(self.shared_linear1(x))
        x = self.dropout(x)
        
        return x

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
        weights = self.shared_layers(data)
        
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

        return action, probs.log_prob(action), probs.entropy(), value
