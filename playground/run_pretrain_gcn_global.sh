#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -t 06:00:00
#SBATCH -o /scratch/mk8737/rl_pretrain/tmp_slurm_%j.out
#SBATCH -e /scratch/mk8737/rl_pretrain/tmp_slurm_%j.err
#SBATCH --mem=64gb

module purge
module load cuda/11.8.0
# SBATCH -p nvidia

# Make sure Conda is initialised for non-interactive shells
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate rl_pretrain

# Copy outputs into designated directory
SCRIPT_NAME="run_pretrain_GCN_global" # all gcn are trained with dataloader
RESULTS_DIR=$RL_PRETRAIN_SCRATCH/results/${SCRIPT_NAME}_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

# Copy code into scratch
#cp -r ~/capstone/GNN_RL_Pretrain/* $SCRATCH_DIR
#cd $SCRATCH_DIR

# Run training
# Pass all arguments from the bash script to the Python script
# if [ $# -ge 1 ]; then
#     RUN_NAME=$1
#     echo "Using provided run name: $RUN_NAME"
#     python pretrain_GAT_network_with_dataloader.py --name "foo"
# else
#     echo "No run name provided, using default"
#     python pretrain_GAT_network_with_dataloader.py
# fi
# python pretrain_GAT_network_with_dataloader.py --name "GAT_dataloader_test_workers8_epochs20"
NUM_WORKERS=4
NUM_EPOCHS=2
POOL_TYPE="global"
# RUN_NAME="GCN_global_workers${NUM_WORKERS}_epochs${NUM_EPOCHS}_cpu"
RUN_NAME="GCN_${POOL_TYPE}_W${NUM_WORKERS}_EP${NUM_EPOCHS}_cpu_${SLURM_JOB_ID}"
python pretrain_GCN_network.py --name $RUN_NAME --pool-type $POOL_TYPE --num-epochs $NUM_EPOCHS --num-workers $NUM_WORKERS

# pretrain_GCN_network.py: error: unrecognized arguments: --batch_size=256 --hidden_size=128 --lr=0.002903850037194237 --num_epochs=500 --num_gcn_layers=4

mv $RL_PRETRAIN_SCRATCH/tmp_slurm_${SLURM_JOB_ID}.out $RESULTS_DIR/job_${SLURM_JOB_ID}.out
mv $RL_PRETRAIN_SCRATCH/tmp_slurm_${SLURM_JOB_ID}.err $RESULTS_DIR/job_${SLURM_JOB_ID}.err

# Save results (logs, model checkpoints, etc.)
#mkdir -p ~/capstone/GNN_RL_Pretrain/results/testing_slurm_$SLURM_JOB_ID
#cp * ~/capstone/GNN_RL_Pretrain/results/testing_slurm_$SLURM_JOB_ID
#cp *.pt ~/capstone/GNN_RL_Pretrain/results_$SLURM_JOB_ID/
