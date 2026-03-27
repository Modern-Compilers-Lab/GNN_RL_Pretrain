## Introduction

- This repo contains code to run experiments with different GNN models on the PEARL pretraining task.

- The neural_nets module implements GAT, GIN, and GCN. GIN and GCN are modular to allow switching pooling mechanisms. As for GAT, it's the original GAT implementation that was used in the paper. Pooling options are defined in pooling.py. Working implementations are present for Global Pooling and SAG Pooling.

- Pretraining scripts are in the repo root, named pretrain_[MODEL_TYPE]_network.py. These scripts are largely duplicates with minor training differences per architecture. Most up to date:
  - GCN: pretrain_GCN_network.py
  - GIN: pretrain_GCN_sag_network.py, pretrain_GIN_network.py
  - GAT: pretrain_GAT_network_with_dataloader.py

- Bash scripts for running pretraining and W&B sweeps are in the playground/ subdirectory. W&B is used for hyperparameter tuning sweeps. MLflow is used for individual runs. This can be changed in the pretraining scripts as needed.

- As currently configured:
  - results/ stores .out and .err from HPC runs
  - sweep_results/ stores .out and .err from W&B sweeps on the HPC
  - saved_weights/ stores model weights
  - wandb/ contains run metadata for wandb sweeps
  - mlruns/ contains run metadata for runs not using W&B

This mostly covers what’s needed to add new architectures, update or create pretraining scripts, and run hyperparameter sweeps. README_OLD.md contains some general information related to the repo, but some of the information in it might be outdated and inaccurate.

## License

This project is released under the same license as the original Modern-Compilers-Lab/GNN_RL_Pretrain repository.

Modifications in this fork are provided under the same terms.
