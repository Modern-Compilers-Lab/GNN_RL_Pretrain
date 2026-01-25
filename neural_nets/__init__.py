"""
This package contains different implementations of neural networks for the Agent in the PEARL RL-based autoschedular.

The models are compared on the pretraining task of predicting execution times.

Models:
- GAT: this is the original Graph Attention Network implementation
- GCN: Graph Convolutional Network implementation
- GIN: Graph Isomorphism Network implementation

Pooling layers used:
- GlobalPooling: global mean/max pooling
- SAGPoolLayer: Self-Attention Graph Pooling
- DiffPoolLayer: Differentiable Pooling -- NEEDS TO BE DEBUGGED

Both models are designed for the same task:
- Input: Program dependency graphs with 175-dimensional node features
- Output: Policy (action probabilities) and value (execution time estimate)
- Parameter count: ~207k-217k parameters for fair comparison
"""

__version__ = "1.0.0"
__author__ = "RL Pretrain Team"

from .gat_model import GAT
from .gcn_model import GCN
from .gin_model import GIN

__all__ = ["GAT", "GCN", "GIN"]