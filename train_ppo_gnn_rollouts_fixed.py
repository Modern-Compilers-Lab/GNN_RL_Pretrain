from config.config import Config

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

from pretrain.embedding import get_embedding_size
from pretrain.lstm_autoencoder_modeling import encoder
from agent.policy_value_nn import GAT

from agent.rollout_worker import RolloutWorker, Transition
from utils.dataset_actor.dataset_actor import DatasetActor

def async_collect_rollouts(rollout_workers, ppo_agent, device, batch_size):
    """
    Asynchronous rollout collector for distributed RL.
    Collects at least batch_size steps (can overshoot by one trajectory).
    Returns: list of rollout results, rollout time stats, etc.
    """
    num_workers = len(rollout_workers)
    in_flight = []
    results = []
    num_steps = 0
    rollout_timings = []
    batch_start = time.time()

    # Launch one rollout per worker
    for w in rollout_workers:
        in_flight.append(w.rollout.remote(ppo_agent.to(device), device))

    while num_steps < batch_size:
        # Wait for any worker to finish
        ready, in_flight = ray.wait(in_flight, num_returns=1)
        result = ray.get(ready[0])
        results.append(result)
        trajectory_len = len(result["trajectory"])
        num_steps += trajectory_len
        rollout_timings.append(result["rollout_time"])
        worker_id = result.get("worker_id", -1)
        print(f"Worker {worker_id} rollout time: {result['rollout_time']:.2f}s (trajectory_len: {trajectory_len})")

        # Relaunch rollout on the same worker
        # You can use worker_id or rotate through your rollout_workers list if needed
        if 0 <= worker_id < num_workers:
            in_flight.append(rollout_workers[worker_id].rollout.remote(ppo_agent.to(device), device))
        else:
            # fallback: just round-robin
            w = rollout_workers[len(results) % num_workers]
            in_flight.append(w.rollout.remote(ppo_agent.to(device), device))

    batch_total_time = time.time() - batch_start
    print(f"Total async collection time: {batch_total_time:.2f}s")
    print(f"Rollout times per trajectory: {[f'{x:.2f}' for x in rollout_timings]}")
    return results, rollout_timings, batch_total_time

if "__main__" == __name__:
    
    # parser = arg.ArgumentParser()
    # parser.add_argument("--ray-address")
    # parser.add_argument("--device", default="cpu")
    # args = parser.parse_args()

    # # Connect to Ray (use ray-address from SLURM script)
    # if args.ray_address:
    #     print(f"Connecting to Ray at address {args.ray_address}")
    #     ray.init(address=args.ray_address)
    # else:
    #     ray.init()  # Local mode (debug)
        
    
    
    # # parser = arg.ArgumentParser() 


    # NUM_ROLLOUT_WORKERS = args.num_nodes

    # # if NUM_ROLLOUT_WORKERS > 1:
    # #     ray.init("auto")
    # # else:
    # #     ray.init()
    parser = arg.ArgumentParser() 

    parser.add_argument("--num-nodes", default=1, type=int)
    
    experiment_name = "concat_final_hidden_cell_state_pretrained_100k"

    parser.add_argument("--name", type=str, default=experiment_name)

    args = parser.parse_args()

    NUM_ROLLOUT_WORKERS = args.num_nodes

    if NUM_ROLLOUT_WORKERS > 1:
        ray.init("auto")
    else:
        ray.init()


    # Init global config to run the Tiramisu env
    Config.init()

    record = []

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
    tag = "100k"
    Config.config.dataset.tags = [tag]
    dataset_worker = DatasetActor.remote(Config.config.dataset)
    pretrained_model_path = Config.config.dataset.pretrained_model_path
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"TRAINING DEVICE: {device}")

    if Config.config.pretrain.embed_access_matrices:
        input_size = 6 + get_embedding_size(Config.config.pretrain.embedding_type) + 9
    else:
        input_size = 718
    
    # Path to the pretrained model
    if pretrained_model_path is not None:
        ppo_agent = GAT(input_size=input_size, num_heads=4, hidden_size=128, num_outputs=56).to(
            device
        )
    
    optimizer = torch.optim.Adam(
        ppo_agent.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5
    )
    value_loss = nn.MSELoss()

    # Load pretrained weights
    pretrained_weights = torch.load(pretrained_model_path, map_location=device)
    ppo_agent.load_state_dict(
        torch.load(
            pretrained_model_path,
            map_location=torch.device(device)
        ),
    )

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=20, num_gpus=0, scheduling_strategy="SPREAD"
        ).remote(dataset_worker, Config.config, worker_id=i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]

    run_name = args.name

    with mlflow.start_run(
        run_name=run_name,
        # run_id="8f80a3b96ea04676928053f7fd90aa4d"
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
                "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS
            }
        )
        best_performance = 0
        global_steps = 0
        total_num_hits = 0
        for u in range(num_updates):
            start_u = time.time()
            print(f"Update {u+1}/{num_updates}")
            
            # optimizer.param_groups[0]["lr"] = final_lr - (final_lr - start_lr) * np.exp(
            #     -2 * u / num_updates
            # )
            
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] - (lr/(num_updates+100))

            # entropy_coeff = entropy_coeff_finish
            entropy_coeff = entropy_coeff_finish - (
                entropy_coeff_finish - entropy_coeff_start
            ) * np.exp(-10*(global_steps / total_steps))

            num_steps = 0
            b_actions = torch.Tensor([]).to(device)
            b_log_probs = torch.Tensor([]).to(device)
            b_rewards = torch.Tensor([]).to(device)
            b_values = torch.Tensor([]).to(device)
            b_advantages = torch.Tensor([]).to(device)
            b_returns = torch.Tensor([]).to(device)
            b_entropy = torch.Tensor([]).to(device)
            b_actions_mask = torch.Tensor([]).to(device)
            b_states = []
            b_speedups = []
            avg_episode_length = 0
            m = 0

            result_refs = []
            rollout_start_times = {}
            ref_to_worker = {}  # Maps Ray ObjectRef to worker index

            # Initial launch: one rollout per worker
            for i in range(NUM_ROLLOUT_WORKERS):
                obj_ref = rollout_workers[i].rollout.remote(ppo_agent.to("cpu"), "cpu")
                result_refs.append(obj_ref)
                rollout_start_times[obj_ref] = time.time()
                ref_to_worker[obj_ref] = i

            rollout_times = []
            process_times = []
            num_steps = 0

            start_time_batch_size = time.time()
            while num_steps < batch_size:
                # Wait for the first available result
                done_refs, result_refs = ray.wait(result_refs, num_returns=1)
                done_ref = done_refs[0]
                result = ray.get(done_ref)
                worker_idx = ref_to_worker.pop(done_ref)

                # --- Timing ---
                rollout_time = result.get("rollout_time")
                if rollout_time is None:
                    rollout_time = time.time() - rollout_start_times[done_ref]
                rollout_times.append(rollout_time)
                rollout_start_times.pop(done_ref, None)
                
                start_process_time = time.time()
                # --- PPO logic as before ---
                b_speedups.append(math.log(result["speedup"], 4))
                trajectory_len = len(result["trajectory"])
                full_trajectory = Transition(*zip(*result["trajectory"]))
                total_num_hits = total_num_hits + result["num_hits"]
                avg_episode_length = (m * avg_episode_length) / (m + 1) + trajectory_len / (m + 1)
                m += 1
                num_steps += trajectory_len

                actions = torch.Tensor(full_trajectory.action).to(device)
                log_probs = torch.Tensor(full_trajectory.log_prob).to(device)
                rewards = torch.Tensor(full_trajectory.reward).to(device)
                values = torch.Tensor(full_trajectory.value).to(device)
                entropies = torch.Tensor(full_trajectory.entropy).to(device)
                advantages = torch.zeros(trajectory_len).to(device)
                returns = torch.zeros(trajectory_len).to(device)
                states = [None] * trajectory_len

                states[-1] = Data(
                    x=torch.tensor(full_trajectory.state[-1][0], dtype=torch.float32),
                    edge_index=torch.tensor(full_trajectory.state[-1][1], dtype=torch.int).transpose(0, 1).contiguous(),
                )

                advantages[-1] = rewards[-1] - values[-1]

                for t in reversed(range(trajectory_len - 1)):
                    td = rewards[t] + gamma * values[t + 1] - values[t]
                    advantages[t] = td + gamma * lambdaa * advantages[t + 1]
                    states[trajectory_len - 2 - t] = Data(
                        x=torch.tensor(full_trajectory.state[trajectory_len - 2 - t][0], dtype=torch.float32),
                        edge_index=torch.tensor(full_trajectory.state[trajectory_len - 2 - t][1], dtype=torch.int).transpose(0, 1).contiguous(),
                    )

                returns = advantages + values

                b_actions = torch.cat([b_actions, actions]).to(device)
                b_log_probs = torch.cat([b_log_probs, log_probs]).to(device)
                b_advantages = torch.cat([b_advantages, advantages]).to(device)
                b_returns = torch.cat([b_returns, returns]).to(device)
                b_entropy = torch.cat([b_entropy, entropies]).to(device)
                b_states.extend(states)

                if num_steps < batch_size:
                    ray.get(rollout_workers[worker_idx].reset.remote())
                    new_obj_ref = rollout_workers[worker_idx].rollout.remote(ppo_agent.to("cpu"), "cpu")
                    result_refs.append(new_obj_ref)
                    rollout_start_times[new_obj_ref] = time.time()
                    ref_to_worker[new_obj_ref] = worker_idx
                
                end_process_time = time.time()
                process_times.append(end_process_time - start_process_time)
                    
            # Reset all workers as before
            ray.get([w.reset.remote() for w in rollout_workers])
            
            end_time_batch_size = time.time()
            print(f"Batch collection time: {(end_time_batch_size - start_time_batch_size):.2f} seconds")
            # After collection, print timing stats:
            print(f"Rollout times per rollout: {rollout_times}")
            if len(rollout_times) > 1:
                times = [t  for t in rollout_times]
                print("Total  time (sum):", sum(times))

            process_times = [t for t in process_times]
            print("Total process time (sum):", sum(process_times))

            b_speedups = torch.Tensor(b_speedups)
            b_states = Batch.from_data_list(b_states).to(device)
            batch_indices = torch.arange(num_steps).to(device)

            ppo_agent.to(device)
            ppo_agent.train()

            v_loss_mean = 0
            policy_loss_mean = 0
            total_loss_mean = 0
            s = 0

            start_training = time.time()
            for e in range(num_epochs):
                start_e = time.time()
                print(f"Epoch {e+1}/{num_epochs}")
                np.random.shuffle(batch_indices)
                for b in range(0, batch_size, mini_batch_size):
                    start, end = b, b + mini_batch_size
                    rand_ind = batch_indices[start:end]
                    _, new_log_prob, new_entropy, new_value = ppo_agent(
                        Batch.from_data_list(b_states[rand_ind]).to(device),
                        actions_mask=None,
                        action=b_actions[rand_ind],
                    )
                    ratio = new_log_prob - b_log_probs[rand_ind]
                    ratio.exp()

                    clipped_ratio = torch.clamp(
                        ratio, 1 - clip_epsilon, 1 + clip_epsilon
                    )
                    clipped_loss = torch.min(
                        ratio * b_advantages[rand_ind],
                        clipped_ratio * b_advantages[rand_ind],
                    )
                    clip_loss = -clipped_loss.mean()

                    v_loss = value_loss(new_value.reshape(-1), b_returns[rand_ind])

                    ent_loss = new_entropy.mean()
                    loss = clip_loss + value_coeff * v_loss - entropy_coeff * ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ppo_agent.parameters(), max_grad_norm)
                    optimizer.step()

                    v_loss_mean = (v_loss_mean * s) / (s + 1) + v_loss.item() / (s + 1)
                    policy_loss_mean = (policy_loss_mean * s) / (
                        s + 1
                    ) + clip_loss.item() / (s + 1)
                    total_loss_mean = (total_loss_mean * s) / (s + 1) + loss.item() / (
                        s + 1
                    )
                    s += 1
                end_e = time.time()
                # print(f"Epoch Time: {(end_e - start_e):.1f} Seconds")

            global_steps += num_steps

            speedups_mean = b_speedups.mean().item()                
            
            end_training = time.time()
            print(f"Training Time: {(end_training - start_training)/60:.1f} Minutes")
            if best_performance < speedups_mean:
                torch.save(ppo_agent.state_dict(), f"{Config.config.dataset.models_save_path}/model_{run_name}_{u}.pt")
                best_performance = speedups_mean

            infos = {
                "Entropy": b_entropy.mean().item(),
                "Episode Length Mean": avg_episode_length,
                "Policy Loss": policy_loss_mean,
                "Value Loss": v_loss_mean,
                "Total Loss": total_loss_mean,
                "Reward Min": b_speedups.min().item(),
                "Reward Average": speedups_mean,
                "Reward Max": b_speedups.max().item(),
                "total number hits": total_num_hits,
            }
            record.append(infos)
            mlflow.log_metrics(
                infos,
                step=global_steps,
            )
            for k,v in infos.items():
                print(f"{k}: {v:.2f}")
            end_u = time.time()
            print(f"Update Time: {(end_u - start_u)/60:.1f} Minutes")
        mlflow.end_run()

    with open(Config.config.tiramisu.logs_dir + f"/{experiment_name}.json", "w") as f:
        json.dump(record, f, indent=4)

    ray.shutdown()
