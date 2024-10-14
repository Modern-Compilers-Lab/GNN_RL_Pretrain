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
    def __init__(
        self,
        input_size=260,
        hidden_size=64,
        num_heads=4,
        num_outputs=32,
    ):
        super(GAT, self).__init__()

        self.conv_layer1 = GATv2Conv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear1 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )
        self.conv_layer2 = GATv2Conv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            heads=num_heads,
        )
        self.linear2 = Linear(
            in_channels=hidden_size * num_heads,
            out_channels=hidden_size,
        )

        convolutions_layers = [
            self.conv_layer1,
            self.conv_layer2,
        ]
        for convlayer in convolutions_layers:
            for name, param in convlayer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
        linear_layers = [
            self.linear1,
            self.linear2,
        ]
        for linearlayer in linear_layers:
            nn.init.xavier_uniform_(linearlayer.weight)

        self.convs_summarizer = Linear(
            in_channels=4 * hidden_size,
            out_channels=hidden_size * 2,
        )

        self.shared_linear1 = Linear(
            in_channels=hidden_size * 2, out_channels=hidden_size
        )
        nn.init.xavier_uniform_(self.shared_linear1.weight)

        self.π = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, num_outputs), std=0.1),
        )

        self.v = nn.Sequential(
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.SELU(),
            self.init_layer(nn.Linear(hidden_size, 1)),
        )

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def shared_layers(self, data):
        x, edges_index, batch_index = (data.x, data.edge_index, data.batch)

        x = self.conv_layer1(x, edges_index)
        x = nn.functional.selu(self.linear1(x))
        x1 = torch.concat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        x = self.conv_layer2(x, edges_index)
        x = nn.functional.selu(self.linear2(x))
        x2 = torch.concat(
            (global_mean_pool(x, batch_index), global_max_pool(x, batch_index)), dim=-1
        )

        x = torch.concat(
            (
                x1,
                x2,
            ),
            dim=-1,
        )

        x = self.convs_summarizer(x)

        x = nn.functional.selu(self.shared_linear1(x))
        return x

    def forward(self, data, actions_mask=None, action=None):
        weights = self.shared_layers(data)
        logits = self.π(weights)
        if actions_mask != None : 
            logits = logits - actions_mask * 1e8
        probs = Categorical(logits=logits)
        if action == None:
            action = probs.sample()
        value = self.v(weights)
        return action, probs.log_prob(action), probs.entropy(), value
