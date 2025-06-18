# GAT Model Ablation Study

This repository contains the code and data for conducting an ablation study on the Graph Attention Network (GAT) model used in the Tiramisu RL project. The model architecture is implemented in `agent/policy_value_nn.py` and the pretraining script is in `pretrain_GAT_network_with_schedules.py`.

## Model Architecture

The GAT model consists of the following key components:

1. **Input Layer**: Takes graph-structured data with input size of 260 features
2. **GAT Layers**:
   - Two GATv2Conv layers with configurable hidden size and number of attention heads
   - Each GAT layer is followed by a linear layer and SELU activation
   - Uses both mean and max pooling for graph-level features
3. **Policy Head (π)**:
   - Three-layer MLP with SELU activations
   - Outputs action probabilities
4. **Value Head (v)**:
   - Three-layer MLP with SELU activations
   - Outputs state value estimates

Key hyperparameters:
- `input_size`: 260 (default)
- `hidden_size`: 64 (default)
- `num_heads`: 4 (default)
- `num_outputs`: 32 (default)
- `dropout_prob`: 0.1 (default)

## Dataset

The pretraining dataset is stored in `pretrain_dataset_12.5k_fixed_duplicates.pkl`. This dataset contains 12,500 samples of function (with their schedules which gives more then 12.5K ) for training the GAT model.

## Guidelines

## Configuration File

The `config/config.yaml` file contains several important parameters that can be modified for the ablation study. Here are the key sections and parameters to consider:

### Tiramisu and Environment Configuration
```yaml
tiramisu:
    tiramisu_path: "/path/to/your/tiramisu"  # Update with your Tiramisu installation path
    workspace: "/path/to/your/workspace_rollouts/"  # Directory for Tiramisu binary files
    experiment_dir: "/path/to/your/experiment_dir/"  # Directory for experiment results
    logs_dir: "/path/to/your/logs/"  # Directory for logs

env_vars: 
    CXX: "/path/to/your/g++"  # Path to your g++ compiler
    TIRAMISU_ROOT: "/path/to/your/tiramisu"  # Should match tiramisu_path
    CONDA_ENV: "/path/to/your/conda/env"  # Path to your conda environment
    LD_LIBRARY_PATH: "LD_LIBRARY_PATH=/path/to/your/lib:${TIRAMISU_ROOT}/3rdParty/Halide/install/lib64:${TIRAMISU_ROOT}/3rdParty/llvm/build/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib:$LD_LIBRARY_PATH"
```

### Dataset Configuration
```yaml
dataset:
    dataset_format: PICKLE
    cpps_path: /path/to/your/dataset/cpps_12.5k.pkl  # Path to your CPPs file
    dataset_path: /path/to/your/dataset/schedules_full_v2_100k.pkl  # Path to your schedules file
    pretrained_model_path: /path/to/your/pretrained_model.pt  # Path to your pretrained model
    save_path: /path/to/your/experiment_dir/save/  # Directory for updated dataset
    models_save_path: /path/to/your/experiment_dir/models/  # Directory for saved models
    results_save_path: /path/to/your/experiment_dir/results/  # Directory for results
    evaluation_save_path: /path/to/your/experiment_dir/evaluation/  # Directory for evaluation results
    shuffle: True
    seed: 133
    saving_frequency: 1000
    is_benchmark: False
```

### Pretraining Configuration
```yaml
pretrain:
    embed_access_matrices: True
    embedding_type: concat_final_hidden_cell_state  # Options:
    # - final_hidden_state
    # - final_cell_state
    # - concat_final_hidden_cell_state
    # - mean_pooling_output
    # - max_pooling_output
    # - flattened_output
```

### Training Hyperparameters
```yaml
hyperparameters:
    num_updates: 15000
    batch_size: 512
    mini_batch_size: 64
    num_epochs: 4
    clip_epsilon: 0.3
    gamma: 0.99
    lambdaa: 0.95
    value_coeff: 2
    entropy_coeff_start: 0.1
    entropy_coeff_finish: 0
    max_grad_norm: 10
    lr: 0.0001
    start_lr: 0.0001
    final_lr: 0.0001
    weight_decay: 0.0001
```

## Getting Started

1. Install Tiramisu Compiler using [this guide](https://docs.google.com/document/d/1fCnPNd37BByYpAAw5c0Y5Mnswcn9wSKqt2nvb8oBwZY/edit?tab=t.0) (This can be skipped if onlt the pretraining with existing data is performed)

2. Set up a new conda environment and install the required packages:

```bash
# Create and activate a new conda environment
conda create -n gat-study python=3.9
conda activate gat-study

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other required packages
pip install -r requirements.txt
```

The requirements.txt file includes:
- torch and related packages for deep learning
- torch-geometric for graph neural networks
- numpy and scipy for numerical computations
- pandas for data manipulation
- scikit-learn for machine learning utilities
- matplotlib for visualization
- tqdm for progress bars
- pytorch-lightning for training utilities
- mlflow for experiment tracking


