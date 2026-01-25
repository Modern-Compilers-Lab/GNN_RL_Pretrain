#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -t 09:00:00
#SBATCH -o /scratch/mk8737/rl_pretrain/tmp_slurm_%j.out
#SBATCH -e /scratch/mk8737/rl_pretrain/tmp_slurm_%j.err
#SBATCH --mem=64gb

module purge
module load cuda/11.8.0

# Make sure Conda is initialised for non-interactive shells
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate rl_pretrain

# Copy outputs into designated directory
RESULTS_DIR=$RL_PRETRAIN_SCRATCH/results/test_pretrain_GAT_network_with_schedules$SLURM_JOB_ID
mkdir -p $RESULTS_DIR

# Copy code into scratch
#cp -r ~/capstone/GNN_RL_Pretrain/* $SCRATCH_DIR
#cd $SCRATCH_DIR

# Run training
python pretrain_GAT_network_with_schedules.py --name "GAT_nondataloader_epochs200"

mv $RL_PRETRAIN_SCRATCH/tmp_slurm_${SLURM_JOB_ID}.out $RESULTS_DIR/job_${SLURM_JOB_ID}.out
mv $RL_PRETRAIN_SCRATCH/tmp_slurm_${SLURM_JOB_ID}.err $RESULTS_DIR/job_${SLURM_JOB_ID}.err

# Save results (logs, model checkpoints, etc.)
#mkdir -p ~/capstone/GNN_RL_Pretrain/results/testing_slurm_$SLURM_JOB_ID
#cp * ~/capstone/GNN_RL_Pretrain/results/testing_slurm_$SLURM_JOB_ID
#cp *.pt ~/capstone/GNN_RL_Pretrain/results_$SLURM_JOB_ID/
