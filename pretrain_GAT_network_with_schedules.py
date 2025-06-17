from builtins import set
from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data, Batch
import time
import ray
import mlflow
import argparse as arg
import json

import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
import matplotlib.pyplot as plt
from pretrain.embedding import get_embedding_size
from pretrain.lstm_autoencoder_modeling import encoder


from agent.rollout_worker import RolloutWorker, Transition, apply_flattened_action
from utils.dataset_actor.dataset_actor import DatasetActor

import ray
import torch
import torch.nn as nn
import math
from torch_geometric.data import Data
from agent.graph_utils import *
from config.config import Config
from env_api.tiramisu_api import TiramisuEnvAPI
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
# Assuming dataset_worker and data loading logic is already set up
# We will prepare a custom data loader for the pretraining task


import os
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch_geometric.data import Data, Batch
import pandas as pd


def get_action_number(transformation: str) -> Optional[int]:
    """Convert a transformation string to its corresponding action number"""
    match = re.match(r'([IPURTS]\d?)\((.*?)\)', transformation)

    if not match:
        return None
        
    trans_type, params = match.groups()
    params = [p.strip() for p in params.split(',')]

    # Interchange: actions 0-3
    if trans_type == 'I':
        level = int(params[0][1])  # L0 -> 0
        if level < 4:
            return level
            
    # Reversal: actions 4-8
    elif trans_type == 'R':
        level = int(params[0][1])  # L0 -> 0
        if level < 5:
            return level + 4
            
    # Skewing: actions 9-11
    elif trans_type == 'S':
        level = int(params[0][1])  # L0 -> 0
        if level < 3:
            return level + 9
            
    # Parallelization: actions 12-13
    elif trans_type == 'P':
        level = int(params[0][1])  # L0 -> 0
        if level < 2:
            return level + 12
            
    # Tiling: actions 14-49
    elif trans_type == 'T2':
        level = int(params[0][1])  # L0 -> 0
        size_x = int(params[2])
        size_y = int(params[3])
        
        size_mappings = [
            (32, 32, 14),   # 14-17
            (64, 64, 18),   # 18-21
            (128, 128, 22), # 22-25
            (32, 64, 26),   # 26-29
            (32, 128, 30),  # 30-33
            (64, 32, 34),   # 34-37
            (64, 128, 38),  # 38-41
            (128, 32, 42),  # 42-45
            (128, 64, 46),  # 46-49
        ]
        
        for sx, sy, base_action in size_mappings:
            if size_x == sx and size_y == sy:
                return base_action + level
                
    # Unrolling: actions 50-54
    elif trans_type == 'U':
        factor = int(params[1])
        power = 0
        while factor > 1:
            factor //= 2
            power += 1
        return 49 + power if power <= 5 else None
        
    return None

def parse_schedule_to_action_list(schedule_str: str) -> List[int]:
    """
    Parse schedule string and return a list of actions. For different schedules,
    add action 55 but don't repeat common transformations from previous schedules.
    Action 55 is added at the start if first schedule doesn't apply to comp00.
    """
    comp_pattern = r'{(.*?)}:(.*?)(?={|$)'
    transform_pattern = r'([IPURTS]\d?\(.*?\))'
    
    action_list = []
    seen_schedules = []
    prev_transforms = set()
    
    matches = list(re.finditer(comp_pattern, schedule_str))
    if matches and matches[0].groups()[0] != 'comp00':
        action_list.append(55)
    
    for match in matches:
        comp_name, schedule = match.groups()
        
        if schedule in seen_schedules:
            continue
        
        transformations = re.findall(transform_pattern, schedule)
        current_transforms = set(transformations)
        
        if seen_schedules:
            action_list.append(55)
            transformations = [t for t in transformations if t not in prev_transforms]
        
        actions = [get_action_number(t) for t in transformations]
        action_list.extend([a for a in actions if a is not None])
        
        seen_schedules.append(schedule)
        prev_transforms = current_transforms
    
    return action_list
class PretrainDataset:
    def __init__(self, dataset_worker, config, save_path="pretrain_dataset_12.5k_fixed_duplicates.pkl"):
        self.dataset_worker = dataset_worker
        self.data = {}  # Initialize as a dictionary
        self.current_program = None
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=True)
        Config.config = config
        self.save_path = save_path
        self.y_mean = None
        self.y_std = None
        self.collected_programs = set()
    # Normalize and denormalize target values with log transformation
    def log_normalize_y(self,y):
        return np.log(1 + y)

    def log_denormalize_y(self,y_normalized):
        return np.exp(y_normalized) - 1
    
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
        
        comp_pattern = r'{(.*?)}:(.*?)(?={|$)'
        """Prepare and process data."""
        # Try loading saved data
        if self.load_saved_data():
            self.y_mean = np.mean([data['y'] for data in self.data.values()])
            self.y_std = np.std([data['y'] for data in self.data.values()])


            for data in self.data.values():
                data['y'] = float(self.log_normalize_y(data['y']))

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
        
        # Replace the fixed num_functions with dynamic collection
        data_collection_active = True

        total_cases = 0  
        try:
            while data_collection_active:
                start_e = time.time()
                prog_infos = ray.get(self.dataset_worker.get_next_function.remote())
                saved_prog_infos = prog_infos
                end_e = time.time()
                
                # Skip if we can't get valid actions mask
                actions_mask = None
                attempts = 0
                max_attempts = 3
                while not isinstance(actions_mask, np.ndarray) and attempts < max_attempts: 
                    actions_mask = self.tiramisu_api.set_program(*saved_prog_infos)
                    attempts += 1
                
                if attempts == max_attempts:
                    print("Failed to get valid actions mask, skipping program")
                    continue
                    
                self.current_program = prog_infos[0]
                
                # Check for duplicates
                if self.current_program in self.collected_programs:
                    print(f"Duplicate function detected: {self.current_program}. Stopping data collection.")
                    data_collection_active = False
                    break
                
                self.collected_programs.add(self.current_program)
                
                y = self.tiramisu_api.scheduler_service.schedule_object.prog.get_execution_time(
                    "initial_execution", Config.config.machine
                )
                
                for config, execution_time in list(prog_infos[1]['execution_times']['jubail'].items()):
                    if config == "initial_execution":
                        print(f"Initial Configuration: {config}, Execution Time: {execution_time}, Current Program: {self.current_program}")
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
                            
                            self.data[self.current_program] = {
                                "node_feats": node_feats,
                                "edge_index": edge_index,
                                "it_index": it_index,
                                "comp_index": comp_index,
                                "y": y
                            }
                        else:
                            print("Execution time is None. Skipping this data point.")
                    else:
                        print(f"Configuration: {config}, Execution Time: {execution_time}, Current Program: {self.current_program}")
                        
                        # Parse schedule actions
                        actions = parse_schedule_to_action_list(config)
                        
                        if actions.count(55) >= len(self.tiramisu_api.scheduler_service.branches):
                            print(f"Skipping configuration {config} due to excessive next actions.")
                            continue
                        
                        actions_mask = None
                        while not isinstance(actions_mask, np.ndarray): 
                            actions_mask = self.tiramisu_api.set_program(*saved_prog_infos)
                        
                        programs_to_verify = []
                        current_node_feats = np.copy(node_feats)
                        current_edge_index = np.copy(edge_index)
                        current_it_index = np.copy(it_index)
                        for action in actions:
                            if action is not None:
                                (
                                    total_speedup,
                                    current_node_feats,
                                    current_edge_index,
                                    legality,
                                    actions_mask,
                                    done,
                                    num_hits,
                                ) = apply_flattened_action(
                                    self.tiramisu_api,
                                    action,
                                    current_node_feats,
                                    current_edge_index,
                                    it_index,
                                    worker_id="0",
                                )
                                
                                program_with_history = f"{self.current_program}_{self.tiramisu_api.scheduler_service.schedule_object.schedule_str}"
                                
                        self.data[program_with_history] = {
                            "node_feats": current_node_feats,
                            "edge_index": current_edge_index,
                            "it_index": it_index,
                            "comp_index": comp_index,
                            "y": execution_time
                        }
                        programs_to_verify.append(program_with_history)

                        # After applying all actions, verify the final schedule matches the config
                        final_schedule = self.tiramisu_api.scheduler_service.schedule_object.schedule_str
                        if final_schedule != config:
                            # Check for the special case where we need to reapply actions
                            expected_parts = list(re.finditer(comp_pattern, config))
                            generated_parts = list(re.finditer(comp_pattern, final_schedule))
                            
                            # Case 1: Same transformation needs to be repeated on multiple computations
                            repeat_transformation = (len(generated_parts) == 1 and 
                                                len(expected_parts) > 1 and
                                                all(part.group(2) == expected_parts[0].group(2) for part in expected_parts) and
                                                generated_parts[0].group(2) == expected_parts[0].group(2))
                            
                            # Case 2: Same transformation but wrong computation
                            wrong_computation = (len(generated_parts) == 1 and 
                                                len(expected_parts) == 1 and
                                                generated_parts[0].group(2) == expected_parts[0].group(2) and
                                                generated_parts[0].group(1) != expected_parts[0].group(1))
                            
                            if repeat_transformation or wrong_computation:
                                print(f"Detected special case. Attempting to fix schedule...")
                                print(f"Expected: {config}")
                                print(f"Current: {final_schedule}")
                                                        
                                # Add next_computation action (55) and reapply the transformation
                                actions_mask = None
                                while not isinstance(actions_mask, np.ndarray):
                                    actions_mask = self.tiramisu_api.set_program(*saved_prog_infos)
                                
                                if repeat_transformation:
                                    current_node_feats = np.copy(node_feats)
                                    current_edge_index = np.copy(edge_index)
                                    # First apply all actions up to current point
                                    for action in actions:  # Apply all but the last action
                                        if action is not None:
                                            (
                                                total_speedup,
                                                current_node_feats,
                                                current_edge_index,
                                                legality,
                                                actions_mask,
                                                done,
                                                num_hits,
                                            ) = apply_flattened_action(
                                                self.tiramisu_api,
                                                action,
                                                np.copy(current_node_feats),
                                                np.copy(current_edge_index),
                                                it_index,
                                                worker_id="0",
                                            )
                                
                                # Apply next_computation action (55)
                                (
                                    total_speedup,
                                    current_node_feats,
                                    current_edge_index,
                                    legality,
                                    actions_mask,
                                    done,
                                    num_hits,
                                ) = apply_flattened_action(
                                    self.tiramisu_api,
                                    55,  # next_computation action
                                    np.copy(current_node_feats),
                                    np.copy(current_edge_index),
                                    it_index,
                                    worker_id="0",
                                )
                                
                                if repeat_transformation:
                                    # Reapply the last transformation
                                    (
                                        total_speedup,
                                        current_node_feats,
                                        current_edge_index,
                                        legality,
                                        actions_mask,
                                        done,
                                        num_hits,
                                    ) = apply_flattened_action(
                                        self.tiramisu_api,
                                        actions[-1],  # Reapply the last action
                                        np.copy(current_node_feats),
                                        np.copy(current_edge_index),
                                        it_index,
                                        worker_id="0",
                                    )
                                if wrong_computation:
                                    # First apply all actions up to current point
                                    for action in actions:  # Apply all but the last action
                                        if action is not None:
                                            (
                                                total_speedup,
                                                current_node_feats,
                                                current_edge_index,
                                                legality,
                                                actions_mask,
                                                done,
                                                num_hits,
                                            ) = apply_flattened_action(
                                                self.tiramisu_api,
                                                action,
                                                np.copy(current_node_feats),
                                                np.copy(current_edge_index),
                                                it_index,
                                                worker_id="0",
                                            )
                                    
                                
                                # Verify the final schedule matches the expected
                                final_schedule = self.tiramisu_api.scheduler_service.schedule_object.schedule_str
                                if final_schedule != config:
                                    print(f"Warning: Generated schedule still doesn't match expected config!")
                                    print(f"Expected: {config}")
                                    print(f"Generated: {final_schedule}")
                                    with open("./schedule_mismatch_log.txt", "a") as log_file:
                                        log_file.write(f"Expected: {config}\n")
                                        log_file.write(f"Generated: {final_schedule}\n")
                                    
                                    total_cases += 1
                                    for program_id in programs_to_verify:
                                        if program_id in self.data:
                                            del self.data[program_id]
                                            print(f"Removed invalid data point: {program_id}")
                            else:
                                # Handle original mismatch case
                                print(f"Warning: Generated schedule doesn't match expected config!")
                                print(f"Expected: {config}")
                                print(f"Generated: {final_schedule}")
                                with open("./schedule_mismatch_log.txt", "a") as log_file:
                                    log_file.write(f"Expected: {config}\n")
                                    log_file.write(f"Generated: {final_schedule}\n")
                                
                                total_cases += 1
                                for program_id in programs_to_verify:
                                    if program_id in self.data:
                                        del self.data[program_id]
                                        print(f"Removed invalid data point: {program_id}")
                                                    
                                    
        except (KeyboardInterrupt, Exception) as e:
            print(f"Data collection stopped: {str(e)}")
            print(f"Total programs collected: {len(self.data)}")
        
        print(f"Total unique programs collected: {len(self.collected_programs)}")
        print(f"Total programs collected: {len(self.data)}")
        print(f"Total cases where generated schedule doesn't match expected config: {total_cases}")

            
            
        # # Prepare graph data for pretraining
        # num_functions = 22046

        # # pbar = tqdm(total=num_functions - len(self.data))  # Initialize progress bar
   
        # while len(self.data) < num_functions:  # Set this limit based on the number of data samples
        #     start_e = time.time()
        #     prog_infos = ray.get(self.dataset_worker.get_next_function.remote())
        #     saved_prog_infos = prog_infos
        #     end_e = time.time()
        #     actions_mask = None
        #     while not isinstance(actions_mask, np.ndarray): 
        #         actions_mask = self.tiramisu_api.set_program(*saved_prog_infos)
            
        #     self.current_program = prog_infos[0]

        #     for config, execution_time in list(prog_infos[1]['execution_times']['jubail'].items()):
        #         if config == "initial_execution":
        #             print(f" ----------------------- Initial Configuration: {config}, Execution Time: {execution_time}, Current_program: {self.current_program} ---------------------------")
        #             y = self.tiramisu_api.scheduler_service.schedule_object.prog.get_execution_time(
        #                 "initial_execution", Config.config.machine
        #             )
        #             print("Slovers : ", self.tiramisu_api.scheduler_service.schedule_object.prog.schedules_solver)
        #             if y is not None:
        #                 annotations = (
        #                     self.tiramisu_api.scheduler_service.schedule_object.prog.annotations
        #                 )
        #                 # Build graph based on annotations
        #                 node_feats, edge_index, it_index, comp_index = build_graph(
        #                     annotations,
        #                     Config.config.pretrain.embed_access_matrices,
        #                     Config.config.pretrain.embedding_type
        #                 )
                        
        #                 node_feats = focus_on_iterators(
        #                     self.tiramisu_api.scheduler_service.branches[0].common_it,
        #                     node_feats,
        #                     it_index
        #                 )
                        
        #                 saved_node_feats, saved_edge_index, saved_it_index, saved_comp_index = node_feats, edge_index, it_index, comp_index
        #                 self.data[self.current_program] = {
        #                     "node_feats": node_feats,
        #                     "edge_index": edge_index,
        #                     "it_index": it_index,
        #                     "comp_index": comp_index,
        #                     "y": y
        #                 }
        #             else:
        #                 num_functions-=1
        #                 print("Execution time is None. Skipping this data point.")
        #         else:
        #             # print(f"Configuration: {config}, Execution Time: {execution_time}, Current_program: {self.current_program}")
        #             # Parse the schedule string to get the actions
        #             actions = parse_schedule_to_action_list(config)
        #             actions_mask = None
        #             while not isinstance(actions_mask, np.ndarray): 
        #                 actions_mask = self.tiramisu_api.set_program(*saved_prog_infos)
                    
        #             # print(f"\t \t Actions: {actions}")
        #             self.current_program = saved_prog_infos[0]
                    
        #             # Track programs created during this action sequence
        #             programs_to_verify = []
                    
        #             for action in actions:
        #                 if action is not None:            
                            
        #                     (
        #                         total_speedup,
        #                         new_node_feats,
        #                         new_edge_index,
        #                         legality,
        #                         actions_mask,
        #                         done,
        #                         num_hits,
        #                     ) = apply_flattened_action(
        #                         self.tiramisu_api,
        #                         action,
        #                         np.copy(saved_node_feats),
        #                         np.copy(saved_edge_index),
        #                         saved_it_index,
        #                         worker_id="0",
        #                     )
        #                     # print(f"\t \t \t Actions Sequence So far : {self.tiramisu_api.scheduler_service.schedule_object.schedule_str}")
                            
        #                     # Create the program identifier with full action history
        #                     program_with_history = f"{self.current_program}_{self.tiramisu_api.scheduler_service.schedule_object.schedule_str}"
                            
        #                     self.data[program_with_history] = {
        #                         "node_feats": new_node_feats,
        #                         "edge_index": new_edge_index,
        #                         "it_index": saved_it_index,
        #                         "comp_index": saved_comp_index,
        #                         "y": y
        #                     }
                            
        #                     programs_to_verify.append(program_with_history)
                    
        #             # After applying all actions, verify the final schedule matches the config
        #             final_schedule = self.tiramisu_api.scheduler_service.schedule_object.schedule_str
        #             if final_schedule != config:
        #                 print(f"Warning: Generated schedule doesn't match expected config!")
        #                 print(f"\t \t Actions Applied: {actions}")
        #                 print(f"Expected: {config}")
        #                 print(f"Generated: {final_schedule}")
                        
        #                 # Remove all data points created during this action sequence
        #                 for program_id in programs_to_verify:
        #                     if program_id in self.data:
        #                         del self.data[program_id]
        #                         print(f"Removed invalid data point: {program_id}")
                                
                # pbar.update(1)
            # print(len(self.data))
            
        # pbar.close()  # Close progress bar after loop finishes
        print("finshed")

        # tiramisu_program_dict = (
        #     self.tiramisu_api.get_current_tiramisu_program_dict()
        # )
        # for current_program in self.data:
        #     self.dataset_worker.update_dataset.remote(
        #         current_program, tiramisu_program_dict
        #     )
        
        
        # Save the data after processing
        self.save_data()

        # Calculate mean and standard deviation for normalization
        self.y_mean = np.mean([data['y'] for data in self.data.values()])
        self.y_std = np.std([data['y'] for data in self.data.values()])


        # Normalize target values
        for data in self.data.values():
            data['y'] = float(self.log_normalize_y(data['y']))
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


    # #Spliting data with 95th percentile clipping
    # def split_data(self, val_split, test_split):
    #     """Split data into training, validation, and test sets with 95th percentile clipping."""
    #     data_items = list(self.data.values())  # Get all data as a list
        
    #     # Split into train/val and test sets
    #     train_val_data, test_data = train_test_split(data_items, test_size=test_split, random_state=42)
    #     train_data, val_data = train_test_split(train_val_data, test_size=val_split / (1 - test_split), random_state=42)

    #     # Extract execution times (y values) for each split
    #     train_y = np.array([data.y for data in train_data])
    #     val_y = np.array([data.y for data in val_data])
    #     test_y = np.array([data.yy for data in test_data])

    #     # Compute 95th percentile thresholds
    #     train_threshold = np.percentile(train_y, 95)
    #     val_threshold = np.percentile(val_y, 95)
    #     test_threshold = np.percentile(test_y, 95)

    #     # Apply clipping
    #     for data in train_data:
    #         data.y = min(data.y, train_threshold)
    #     for data in val_data:
    #         data.y = min(data.y, val_threshold)
    #     for data in test_data:
    #         data.y = min(data.y, test_threshold)

    #     # Store the clipped data
    #     self.train_data = train_data
    #     self.val_data = val_data
    #     self.test_data = test_data

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
    model, dataset_worker, device, config, num_epochs=1000, batch_size=128, lr=1e-3
):
     # L2 Regularization (Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = PretrainDataset(dataset_worker, config)
    
    print("finidhed preparing data")
    dataset.prepare_data()

    best_val_loss = float("inf")
    print("finidhed preparing data")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_batches = len(dataset.train_data) // batch_size
        print("total_batches", total_batches)
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
            for batch_idx in range(total_val_batches):
                batch = dataset.get_batch("val", batch_size).to(device)

                # Pass through shared layers
                weights = model.shared_layers(batch)

                # Value prediction through the value layers
                value_preds = model.v(weights).squeeze(-1)

                # Use value predictions to estimate execution time
                execution_time_preds = value_preds

                # Compute validation loss
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

        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pretrained_model_12.5k_L2_Regularization_GAT_512.pt")

    # Testing phase
    model.load_state_dict(torch.load("pretrained_model_12.5k_L2_Regularization_GAT_512.pt"))
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
            execution_time_preds =  model.v(weights).squeeze(-1)
            
            # Collect predictions and ground truth
            real_times.extend(dataset.log_denormalize_y(batch.y.cpu().numpy()))
            predicted_times.extend(dataset.log_denormalize_y(execution_time_preds.cpu().numpy()))
            
            test_loss = criterion(execution_time_preds, batch.y.to(device))
            total_test_loss += test_loss.item()

    # Calculate average loss
    avg_test_loss = total_test_loss / total_test_batches
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Create the DataFrame
    comparison_df = pd.DataFrame({
        "Real Execution Time": real_times,
        "Predicted Execution Time": predicted_times
    })
    # Add a column for the absolute difference
    comparison_df["Difference"] = (comparison_df["Real Execution Time"] - comparison_df["Predicted Execution Time"])

    # Add a column for the absolute error
    comparison_df["Absolute Error"] = comparison_df["Difference"].abs()

    # Display basic statistics about the differences
    error_stats = comparison_df["Absolute Error"].describe()
    print("Error Statistics:")
    print(error_stats)

    # Optionally, visualize the differences
    plt.figure(figsize=(10, 6))
    plt.hist(comparison_df["Absolute Error"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Absolute Errors", fontsize=14)
    plt.xlabel("Absolute Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig('errors.png')
    plt.show()


    print("Training complete. Final model saved.")



# Example usage
if "__main__" == __name__:
    parser = arg.ArgumentParser() 

    parser.add_argument("--num-nodes", default=1, type=int)
    
    experiment_name = "pretrained_12.5k_L2_3GAT_512"

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
    tag = "12.5k"
    Config.config.dataset.tags = [tag]
    dataset_worker = DatasetActor.remote(Config.config.dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"TRAINING DEVICE: {device}")
    # Initialize GAT model
    if Config.config.pretrain.embed_access_matrices:
        input_size = 6 + get_embedding_size(Config.config.pretrain.embedding_type) + 9
    else:
        input_size = 718
    model = GAT_SCALED(input_size=input_size, hidden_size=128, num_heads=4, num_outputs=56).to(device)

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
        pretrain_model(model, dataset_worker, device, Config.config, num_epochs=3000, batch_size=512, lr=lr)
    
        # Log final model after training
        mlflow.pytorch.log_model(model, "final_gat_model1")
    
    # Save the pretrained model
    torch.save(model.state_dict(), "pretrained_model_12.5k_L2_Regularization_3GAT_512.pt")
