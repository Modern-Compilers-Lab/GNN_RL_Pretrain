import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import time
import ray
import mlflow
import argparse as arg
import json
import math
import numpy as np

import matplotlib.pyplot as plt
from pretrain.embedding import get_embedding_size
from pretrain.lstm_autoencoder_modeling import encoder
from agent.policy_value_nn import GAT

from agent.rollout_worker import RolloutWorker, Transition
from utils.dataset_actor.dataset_actor import DatasetActor

from agent.graph_utils import *
from config.config import Config
from env_api.tiramisu_api import TiramisuEnvAPI
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import os
import pickle
import pandas as pd


class PretrainDataset:
    def __init__(self, dataset_worker, config, save_path="pretrain_dataset_12.5k_dropout.pkl"):
        self.dataset_worker = dataset_worker
        self.data = {}  # Initialize as a dictionary
        self.current_program = None
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=True)
        Config.config = config
        self.save_path = save_path
        self.y_mean = None
        self.y_std = None

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_normalized):
        return y_normalized * self.y_std + self.y_mean

    def load_saved_data(self):
        """Load the dataset from a saved file if it exists."""
        if os.path.exists(self.save_path):
            print(f"Loading dataset from {self.save_path}")
            with open(self.save_path, "rb") as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} data objects.")
            return True
        return False

    def save_data(self):
        """Save the processed dataset to a file."""
        print(f"Saving dataset to {self.save_path}")
        with open(self.save_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Saved {len(self.data)} data objects.")

    def prepare_data(self, val_split=0.1, test_split=0.1):
        """Prepare and process data."""
        # Try loading saved data
        if self.load_saved_data():
            self.y_mean = np.mean([data['y'] for data in self.data.values()])
            self.y_std = np.std([data['y'] for data in self.data.values()])


            for data in self.data.values():
                data['y'] = float(self.normalize_y(data['y']))

            graph_data = {}
            for program_name, data in self.data.items():
                node_feats = data["node_feats"]
                edge_index = data["edge_index"]
                y = data["y"]
                # Create a PyTorch Geometric Data object
                graph_data[program_name] = Data(
                    x=torch.tensor(node_feats, dtype=torch.float32).to(device),
                    edge_index=torch.tensor(edge_index, dtype=torch.long)
                                    .transpose(0, 1)
                                    .contiguous()
                                    .to(device),
                    y = y
                )
            self.data = graph_data
            self.split_data(val_split, test_split)
            return
                
        programs = self.tiramisu_api.dataset_service.cpps_dataset
               
        # Prepare graph data for pretraining
        num_functions = 22046

        pbar = tqdm(total=num_functions - len(self.data))  # Initialize progress bar
   
        while len(self.data) < num_functions:  # Set this limit based on the number of data samples
            
            start_e = time.time()
            prog_infos = ray.get(self.dataset_worker.get_next_function.remote())
            end_e = time.time()
            actions_mask = self.tiramisu_api.set_program(*prog_infos)
            
            self.current_program = prog_infos[0]
            print(prog_infos[1]['execution_times'])
            y = self.tiramisu_api.scheduler_service.schedule_object.prog.get_execution_time(
                "initial_execution", Config.config.machine
            )
            if y is not None:
                annotations = (
                    self.tiramisu_api.scheduler_service.schedule_object.prog.annotations
                )
                # Build graph based on annotations
                node_feats, edge_index, it_index, comp_index = build_graph(
                    annotations,
                    Config.config.pretrain.embed_access_matrices,
                    Config.config.pretrain.embedding_type
                )
                node_feats = focus_on_iterators(
                    self.tiramisu_api.scheduler_service.branches[0].common_it,
                    node_feats,
                    it_index
                )
                
                # Save data in the desired format
                self.data[self.current_program] = {
                    "node_feats": node_feats,
                    "edge_index": edge_index,
                    "it_index": it_index,
                    "comp_index": comp_index,
                    "y": y
                }
                
                pbar.update(1)  # Update progress bar for each iteration
            else:
                num_functions-=1
                print("Execution time is None. Skipping this data point.")
            
        pbar.close()  # Close progress bar after loop finishes
      
        # Save the data after processing
        self.save_data()

        # Calculate mean and standard deviation for normalization
        self.y_mean = np.mean([data['y'] for data in self.data.values()])
        self.y_std = np.std([data['y'] for data in self.data.values()])

        # Normalize target values
        for data in self.data.values():
            data['y'] = float(self.normalize_y(data['y']))
        graph_data = {}
        for program_name, data in self.data.items():
            node_feats = data["node_feats"]
            edge_index = data["edge_index"]
            y = data["y"]
            # Create a PyTorch Geometric Data object
            graph_data[program_name] = Data(
                x=torch.tensor(node_feats, dtype=torch.float32).to(device),
                edge_index=torch.tensor(edge_index, dtype=torch.long)
                                .transpose(0, 1)
                                .contiguous()
                                .to(device),
                y = y
            )
        self.data = graph_data

        # Split data into training, validation, and test sets
        self.split_data(val_split, test_split)

    def split_data(self, val_split, test_split):
        """Split data into training, validation, and test sets."""
        data_items = list(self.data.values())  # Get all data as a list
        train_val_data, test_data = train_test_split(data_items, test_size=test_split, random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=val_split / (1 - test_split), random_state=42)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def get_batch(self, data_split, batch_size):
        """Retrieve a batch of data."""
        if data_split == "train":
            data = self.train_data
        elif data_split == "val":
            data = self.val_data
        elif data_split == "test":
            data = self.test_data
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'.")

        indices = np.random.choice(len(data), batch_size)
        batch_data = [data[i] for i in indices]
        return Batch.from_data_list(batch_data)

def pretrain_model(
    model, dataset_worker, device, config, num_epochs=1500, batch_size=64, lr=1e-3
):
    model.to(device)
    
    # L2 Regularization (Weight Decay)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = PretrainDataset(dataset_worker, config)
    dataset.prepare_data()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_batches = len(dataset.train_data) // batch_size

        for _ in range(total_batches):
            batch = dataset.get_batch("train", batch_size).to(device)
            
            optimizer.zero_grad()

            # Pass through shared layers
            weights = model.shared_layers(batch)

            # Value prediction through the value layers
            value_preds = model.v(weights).squeeze(-1)

            # Execution time prediction = value head output
            execution_time_preds = value_preds

            # Compute loss
            loss = criterion(execution_time_preds, batch.y.to(device))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / total_batches

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_batches = len(dataset.val_data) // batch_size

        with torch.no_grad():
            for _ in range(total_val_batches):
                batch = dataset.get_batch("val", batch_size).to(device)

                weights = model.shared_layers(batch)
                value_preds = model.v(weights).squeeze(-1)
                execution_time_preds = value_preds

                val_loss = criterion(execution_time_preds, batch.y.to(device))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / total_val_batches

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pretrained_model_12.5k_dropout.pt")

    # Testing phase
    model.load_state_dict(torch.load("pretrained_model_12.5k_dropout.pt"))
    model.eval()

    criterion = nn.MSELoss()
    total_test_loss = 0
    total_test_batches = len(dataset.test_data) // batch_size

    real_times = []
    predicted_times = []

    with torch.no_grad():
        for _ in range(total_test_batches):
            batch = dataset.get_batch("test", batch_size).to(device)
            weights = model.shared_layers(batch)
            execution_time_preds = model.v(weights).squeeze(-1)
            
            # Collect predictions and ground truth
            real_times.extend(dataset.denormalize_y(batch.y.cpu().numpy()))
            predicted_times.extend(dataset.denormalize_y(execution_time_preds.cpu().numpy()))
            
            test_loss = criterion(execution_time_preds, batch.y.to(device))
            total_test_loss += test_loss.item()

    # Calculate average loss
    avg_test_loss = total_test_loss / total_test_batches
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("Training complete. Final model saved.")



# Example usage
if "__main__" == __name__:
    parser = arg.ArgumentParser() 

    parser.add_argument("--num-nodes", default=1, type=int)
    
    experiment_name = "pretrain_1500_jubail"

    parser.add_argument("--name", type=str, default=experiment_name)

    args = parser.parse_args()

    # Initialize dataset worker (assuming it's already set up correctly)
    record = []
    Config.init()
    # Hyperparameters
    num_updates = Config.config.hyperparameters.num_updates
    batch_size = Config.config.hyperparameters.batch_size
    mini_batch_size = Config.config.hyperparameters.mini_batch_size
    num_epochs = Config.config.hyperparameters.num_epochs
    total_steps = num_updates * batch_size
    
    clip_epsilon = Config.config.hyperparameters.clip_epsilon
    gamma = Config.config.hyperparameters.gamma
    lambdaa = Config.config.hyperparameters.lambdaa
    
    value_coeff = Config.config.hyperparameters.value_coeff
    entropy_coeff_start = Config.config.hyperparameters.entropy_coeff_start
    entropy_coeff_finish = Config.config.hyperparameters.entropy_coeff_finish
    max_grad_norm = Config.config.hyperparameters.max_grad_norm
    lr = Config.config.hyperparameters.lr
    start_lr = Config.config.hyperparameters.start_lr
    final_lr = Config.config.hyperparameters.final_lr
    weight_decay = Config.config.hyperparameters.weight_decay
    tag = "full"
    Config.config.dataset.tags = [tag]
    dataset_worker = DatasetActor.remote(Config.config.dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"TRAINING DEVICE: {device}")
    # Initialize GAT model
    if Config.config.pretrain.embed_access_matrices:
        input_size = 6 + get_embedding_size(Config.config.pretrain.embedding_type) + 9
    else:
        input_size = 718
    print(input_size)
    model = GAT(input_size=input_size, hidden_size=128, num_heads=4, num_outputs=56).to(device)

    # Pretrain the model
    run_name = args.name

    with mlflow.start_run(
        run_name=run_name,
    ) as run:
        mlflow.log_params(
            {
                "total_steps": total_steps,
                "num_updates": num_updates,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "mini_batch_size": mini_batch_size,
                "lr": lr,
                "gamma": gamma,
                "lambdaa": lambdaa,
                "weight_decay": weight_decay,
                "clip_epsilon": clip_epsilon,
                "max_grad_norm": max_grad_norm,
                "value_coeff": value_coeff,
                "entropy_coeff_start": entropy_coeff_start,
                "entropy_coeff_finish": entropy_coeff_finish,
            }
        )
        pretrain_model(model, dataset_worker, device, Config.config, num_epochs=1500, batch_size=64, lr=lr)
    
        # Log final model after training
        mlflow.pytorch.log_model(model, "pretrained_model_12.5k_dropout")
    
    # Save the pretrained model
    torch.save(model.state_dict(), "pretrained_model_12.5k_dropout.pt")
