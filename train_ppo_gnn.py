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

if "__main__" == __name__:
    parser = arg.ArgumentParser() 

    parser.add_argument("--num-nodes", default=1, type=int)
    
    experiment_name = "final_hidden_state_u500_b500_ent0.5"

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

    dataset_worker = DatasetActor.remote(Config.config.dataset)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"TRAINING DEVICE: {device}")

    if Config.config.pretrain.embed_access_matrices:
        input_size = 6 + get_embedding_size(Config.config.pretrain.embedding_type) + 9
    else:
        input_size = 718
    
    ppo_agent = GAT(input_size=input_size, num_heads=4, hidden_size=128, num_outputs=56).to(
        device
    )
    
    optimizer = torch.optim.Adam(
        ppo_agent.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5
    )
    value_loss = nn.MSELoss()

    # ppo_agent.load_state_dict(
    #     torch.load(
    #         f"{Config.config.dataset.models_save_path}/model_experiment_101_239.pt",
    #         map_location=torch.device(device)
    #     ),
    # )

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=12, num_gpus=1, scheduling_strategy="SPREAD"
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

            while num_steps < batch_size:
                results = ray.get(
                    [
                        rollout_workers[i].rollout.remote(ppo_agent.to("cpu"), "cpu")
                        for i in range(NUM_ROLLOUT_WORKERS)
                    ]
                )

                for result in results:
                    b_speedups.append(math.log(result["speedup"], 4))
                    trajectory_len = len(result["trajectory"])
                    full_trajectory = Transition(*zip(*result["trajectory"]))
                    avg_episode_length = (m * avg_episode_length) / (
                        m + 1
                    ) + trajectory_len / (m + 1)
                    m += 1
                    num_steps += trajectory_len

                    actions = torch.Tensor(full_trajectory.action).to(device)
                    log_probs = torch.Tensor(full_trajectory.log_prob).to(device)
                    rewards = torch.Tensor(full_trajectory.reward).to(device)
                    values = torch.Tensor(full_trajectory.value).to(device)
                    entropies = torch.Tensor(full_trajectory.entropy).to(device)
                    # actions_mask = torch.Tensor(full_trajectory.actions_mask).to(device)
                    # Calculating advantages and lambda returns
                    advantages = torch.zeros(trajectory_len).to(device)
                    returns = torch.zeros(trajectory_len).to(device)

                    states = [None] * trajectory_len

                    states[-1] = Data(
                        x=torch.tensor(
                            full_trajectory.state[-1][0], dtype=torch.float32
                        ),
                        edge_index=torch.tensor(
                            full_trajectory.state[-1][1], dtype=torch.int
                        )
                        .transpose(0, 1)
                        .contiguous(),
                    )

                    advantages[-1] = rewards[-1] - values[-1]

                    for t in reversed(range(trajectory_len - 1)):
                        td = rewards[t] + gamma * values[t + 1] - values[t]
                        advantages[t] = td + gamma * lambdaa * advantages[t + 1]
                        states[trajectory_len - 2 - t] = Data(
                            x=torch.tensor(
                                full_trajectory.state[trajectory_len - 2 - t][0],
                                dtype=torch.float32,
                            ),
                            edge_index=torch.tensor(
                                full_trajectory.state[trajectory_len - 2 - t][1],
                                dtype=torch.int,
                            )
                            .transpose(0, 1)
                            .contiguous(),
                        )

                    returns = advantages + values

                    b_actions = torch.cat([b_actions, actions]).to(device)
                    b_log_probs = torch.cat([b_log_probs, log_probs]).to(device)
                    b_advantages = torch.cat([b_advantages, advantages]).to(device)
                    b_returns = torch.cat([b_returns, returns]).to(device)
                    b_entropy = torch.cat([b_entropy, entropies]).to(device)
                    # b_actions_mask = torch.cat([b_actions_mask, actions_mask]).to(device)
                    b_states.extend(states)

                ray.get(
                    [
                        rollout_workers[i].reset.remote()
                        for i in range(NUM_ROLLOUT_WORKERS)
                    ]
                )

            b_speedups = torch.Tensor(b_speedups)
            b_states = Batch.from_data_list(b_states).to(device)
            batch_indices = torch.arange(num_steps).to(device)

            ppo_agent.to(device)
            ppo_agent.train()

            v_loss_mean = 0
            policy_loss_mean = 0
            total_loss_mean = 0

            s = 0

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
