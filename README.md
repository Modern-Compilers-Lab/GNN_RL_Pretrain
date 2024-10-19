# Context
The project aims to perform Automatic Code Optimization using reinforcement learning in the Tiramisu compiler. We investigate the use of pretrained autoencoders to get the program's access matrices embeddings, which reduces the size of computational vectors.

# Project Setup
## 1. Installing Tiramisu
Here is a detailed and updated [guide](https://tremendous-radio-85b.notion.site/Tiramisu-Installation-Guide-3f5eed16d5d641bb8ed45c493ca2ca8e) on installing Tiramisu compiler.

## 2. Running RL Agent
Here is a detailed [guide](https://kamel-brouthen.notion.site/Running-RL-Agent-07df4e9c793b44e7a5d30ed057f2b079https://kamel-brouthen.notion.site/Running-RL-Agent-07df4e9c793b44e7a5d30ed057f2b079) on running the RL agent. Additionally, the guide includes steps to set up Cuda and PyTorch for GPU training.

## 3. Codebase
The project is based on the following work:

### 3.1 Reinforcement Learning Agent applying Proximal Policy Optimization (PPO) with a Graph Neural Network (GNN) backbone
- Repository: [gnn_rl](https://github.com/Tiramisu-Compiler/gnn_rl) by [Lamouri Djamel Rassem](https://github.com/djamelrassem)

### 3.2 Pretrained Autoencoder for Tiramisu Program's Computational Vector
- Repository: [cost_model_pretrain](https://github.com/Tiramisu-Compiler/cost_model_pretrain) by Chunting Liu

## 4. Training Workflow
- Create conda environment
```shell
conda env create -f environment.yaml
```
- Activate conda environment
```shell
conda activate tiramisu-build-env
```
- Train the agent
```shell
python train_ppo_gnn.py 
```
- Evaluate the agent
 ```shell
python evaluate_ppo_gnn.py 
```

# 5. Resources
- [Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code](https://arxiv.org/abs/1804.10694)
- [A Deep Learning Based Cost Model for Automatic Code Optimization](https://arxiv.org/abs/2104.04955https://arxiv.org/abs/2104.04955)
- [LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers](https://arxiv.org/abs/2403.11522)
- [Utilisation d’apprentissage par renforcement pour l’optimisation automatique de code dans TiramisuUtilisation d’apprentissage par renforcement pour l’optimisation automatique de code dans Tiramisu](https://www.researchgate.net/publication/372128690_Utilisation_d'apprentissage_par_renforcement_pour_l'optimisation_automatique_de_code_dans_Tiramisu)
- [Automatic Code Optimization in the MLIR Compiler Using Deep Reinforcement Learning](https://www.researchgate.net/publication/382047058_Automatic_Code_Optimization_in_the_MLIR_Compiler_Using_Deep_Reinforcement_Learning)