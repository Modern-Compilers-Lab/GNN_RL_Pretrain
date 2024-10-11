import os
import ray
import json
from config.config import Config

import torch
from agent.policy_value_nn import GAT
from agent.rollout_worker import RolloutWorker
from utils.dataset_actor.dataset_actor import DatasetActor
from env_api.core.services.compiling_service import CompilingService

from pretrain.embedding import get_embedding_size

NUM_ROLLOUT_WORKERS = 1

def write_cpp_file(schedule_object):
    tiramisu_prog = schedule_object.prog
    optim_list = schedule_object.schedule_list
    cpp_code = CompilingService.get_schedule_code(
        tiramisu_program=tiramisu_prog, optims_list=optim_list
    )
    CompilingService.write_cpp_code(
        cpp_code,
        os.path.join(
            Config.config.tiramisu.experiment_dir,
            "evaluation",
            schedule_object.prog.name,
        ),
    )

if "__main__" == __name__:

    full_log = ""

    # ray.init("auto")
    ray.init()
    # Init global config to run the Tiramisu env
    Config.init()

    dataset_worker = DatasetActor.remote(Config.config.dataset)
    # device = "cpu"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if Config.config.pretrain.embed_access_matrices:
        input_size = 6 + get_embedding_size(Config.config.pretrain.embedding_type) + 9
    else:
        input_size = 718
    
    ppo_agent = GAT(input_size=input_size, num_heads=4, hidden_size=128, num_outputs=56).to(
        device
    )
    
    model_name = 'model_full_computational_vector_u500_b500_ent0.5_426'
    ppo_agent.load_state_dict(
        torch.load(
            f"{Config.config.dataset.models_save_path}/{model_name}.pt",
            map_location=torch.device(device),
        )
    )

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=12, num_gpus=1, scheduling_strategy="SPREAD"
        ).remote(dataset_worker, Config.config, worker_id=i)
        for i in range(NUM_ROLLOUT_WORKERS)
    ]

    num_functions = ray.get(dataset_worker.get_dataset_size.remote())

    res = {}

    for _ in range(num_functions // NUM_ROLLOUT_WORKERS):
        results = ray.get(
            [
                rollout_workers[i].rollout.remote(ppo_agent, "cpu")
                for i in range(NUM_ROLLOUT_WORKERS)
            ]
        )
        for result in results:
            full_log += result["log_trajectory"]
            res[result["schedule_object"].prog.name] = {}
            res[result["schedule_object"].prog.name]["schedule"] = result[
                "schedule_object"
            ].schedule_str
            res[result["schedule_object"].prog.name]["speedup"] = result["speedup"]
            write_cpp_file(result["schedule_object"])

        ray.get([rollout_workers[i].reset.remote() for i in range(NUM_ROLLOUT_WORKERS)])

    results = ray.get(
        [
            rollout_workers[i].rollout.remote(ppo_agent, "cpu")
            for i in range(num_functions % NUM_ROLLOUT_WORKERS)
        ]
    )
    for result in results:
        full_log += result["log_trajectory"]
        res[result["schedule_object"].prog.name] = {}
        res[result["schedule_object"].prog.name]["schedule"] = result[
            "schedule_object"
        ].schedule_str
        res[result["schedule_object"].prog.name]["speedup"] = result["speedup"]
        write_cpp_file(result["schedule_object"])

    ray.get(
        [
            rollout_workers[i].reset.remote()
            for i in range(num_functions % NUM_ROLLOUT_WORKERS)
        ]
    )

    os.makedirs(f"{Config.config.dataset.results_save_path}" + f"{model_name}", exist_ok=True)

    with open(f"{Config.config.dataset.results_save_path}" + f"{model_name}/results.json", "w") as file:
        json.dump(res, file, indent=4)

    with open(f"{Config.config.dataset.results_save_path}" + f"/{model_name}/log.txt", "w") as file:
        file.write(full_log)

    ray.shutdown()
