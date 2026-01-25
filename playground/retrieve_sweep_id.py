import os
import sys
import yaml
import wandb
from contextlib import redirect_stderr, redirect_stdout

# Load the sweep config
with open("playground/gin_sag_sweep.yaml") as f: # edit sweep config path if needed
    sweep_config = yaml.safe_load(f)

# Create the sweep, redirecting stderr to /dev/null to silence it
with open(os.devnull, 'w') as devnull:
    with redirect_stdout(devnull), redirect_stderr(devnull):
        sweep_id = wandb.sweep(sweep=sweep_config, project="rl_pretrain_hp_sweeps", entity="mk8737-new-york-university")

# Print the sweep ID to stdout
print(sweep_id)