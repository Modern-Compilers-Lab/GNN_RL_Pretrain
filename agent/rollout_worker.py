from collections import namedtuple

import ray
import torch
import torch.nn as nn
import math
from torch_geometric.data import Data
from agent.graph_utils import *
from config.config import Config
from env_api.tiramisu_api import TiramisuEnvAPI


def apply_flattened_action(
    tiramisu_api: TiramisuEnvAPI,
    action,
    node_feats,
    edge_index,
    it_index,
    worker_id="0",
):
    done = False
    if action < 4:
        loop_level = action
        # Interchange of loops (0,1) (1,2) (2,3) (3,4)
        (
            speedup,
            legality,
            actions_mask,
        ) = tiramisu_api.interchange(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_interchange(
                [its[loop_level], its[loop_level + 1]], edge_index, it_index
            )
    elif action < 9:
        loop_level = action - 4
        # Reversal from 0 to 4
        (
            speedup,
            legality,
            actions_mask,
        ) = tiramisu_api.reverse(
            loop_level=loop_level, env_id=action, worker_id=worker_id
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_reversal(its[loop_level], node_feats, it_index)
    elif action < 12:
        loop_level = action - 9
        # Skewing 0,1 to 2,3
        speedup, legality, actions_mask = tiramisu_api.skew(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            env_id=action,
            worker_id=worker_id,
        )

        if legality:
            skewing_params = (
                tiramisu_api.scheduler_service.schedule_object.schedule_list[-1].params[
                    -2:
                ]
            )
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_skewing(
                [its[loop_level], its[loop_level + 1]],
                skewing_params,
                node_feats,
                it_index,
            )
    elif action < 14:
        loop_level = action - 12
        # For parallelization 0 and 1
        (
            speedup,
            legality,
            actions_mask,
        ) = tiramisu_api.parallelize(
            loop_level=loop_level, env_id=action, worker_id=worker_id
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_parallelization(its[loop_level], node_feats, it_index)
    elif action < 18:
        loop_level = action - 14
        size = 32
        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size,
            size_y=size,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size, size], node_feats, it_index
            )
    elif action < 22:
        loop_level = action - 18
        size = 64
        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size,
            size_y=size,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size, size], node_feats, it_index
            )
    elif action < 26:
        loop_level = action - 22
        size = 128
        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size,
            size_y=size,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size, size], node_feats, it_index
            )
    elif action < 30:
        loop_level = action - 26
        size_x = 32
        size_y = 64

        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size_x,
            size_y=size_y,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size_x, size_y], node_feats, it_index
            )
    elif action < 34:
        loop_level = action - 30
        size_x = 32
        size_y = 128

        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size_x,
            size_y=size_y,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size_x, size_y], node_feats, it_index
            )
    elif action < 38:
        loop_level = action - 34
        size_x = 64
        size_y = 32

        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size_x,
            size_y=size_y,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size_x, size_y], node_feats, it_index
            )
    elif action < 42:
        loop_level = action - 38
        size_x = 64
        size_y = 128

        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size_x,
            size_y=size_y,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size_x, size_y], node_feats, it_index
            )
    elif action < 46:
        loop_level = action - 42
        size_x = 128
        size_y = 32

        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size_x,
            size_y=size_y,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size_x, size_y], node_feats, it_index
            )
    elif action < 50:
        loop_level = action - 46
        size_x = 128
        size_y = 64

        speedup, legality, actions_mask = tiramisu_api.tile2D(
            loop_level1=loop_level,
            loop_level2=loop_level + 1,
            size_x=size_x,
            size_y=size_y,
            env_id=action,
            worker_id=worker_id,
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_tiling(
                [its[loop_level], its[loop_level + 1]], [size_x, size_y], node_feats, it_index
            )
    elif action < 55:
        factor = action - 49
        speedup, legality, actions_mask = tiramisu_api.unroll(
            unrolling_factor=2**factor, env_id=action, worker_id=worker_id
        )
        if legality:
            branch = tiramisu_api.scheduler_service.current_branch
            its = tiramisu_api.scheduler_service.branches[branch].common_it
            apply_unrolling(its[-1], 2**factor, node_feats, it_index)
    else:
        # Next case
        next_branch_mask = tiramisu_api.scheduler_service.next_branch()
        if not (isinstance(next_branch_mask, np.ndarray)):
            speedup, legality, actions_mask = (
                1,
                True,
                np.zeros(56),
            )
            done = True
        else:
            speedup, legality, actions_mask = (
                1,
                True,
                next_branch_mask,
            )
            branch = tiramisu_api.scheduler_service.current_branch
            node_feats = focus_on_iterators(
                tiramisu_api.scheduler_service.branches[branch].common_it,
                node_feats,
                it_index,
            )

    return speedup, node_feats, edge_index, legality, actions_mask, done


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "value", "log_prob", "entropy", "actions_mask")
)


@ray.remote
class RolloutWorker:
    def __init__(self, dataset_worker, config, encoder, worker_id=0):

        Config.config = config
        self.tiramisu_api = TiramisuEnvAPI(local_dataset=False)
        self.dataset_worker = dataset_worker

        # Variables related to workers and the environment
        self.worker_id = worker_id
        self.current_program = None

        # Variables related to the RL+Tiramisu train cycle
        self.state = None
        self.actions_mask = None
        self.previous_speedup = None
        self.steps = None
        
        self.encoder = encoder

        # Initializing values and the episode
        self.reset()

    def reset(self):
        actions_mask = None
        while not isinstance(actions_mask, np.ndarray):
            prog_infos = ray.get(self.dataset_worker.get_next_function.remote())
            actions_mask = self.tiramisu_api.set_program(*prog_infos)

        self.current_program = prog_infos[0]

        self.actions_mask = torch.tensor(actions_mask)
        annotations = (
            self.tiramisu_api.scheduler_service.schedule_object.prog.annotations
        )
        node_feats, edge_index, it_index, comp_index = build_graph(annotations, self.encoder)

        node_feats = focus_on_iterators(
            self.tiramisu_api.scheduler_service.branches[0].common_it,
            node_feats,
            it_index,
        )

        self.previous_speedup = 1
        self.steps = 0
        self.state = (node_feats, edge_index, it_index)

    def rollout(self, model: nn.Module, device: str):
        model.to(device)
        model.eval()
        trajectory = []
        done = False
        log_trajectory = "#" * 50
        log_trajectory += f"\nFunction  : {self.current_program}"

        while not done:
            prev_actions_mask = self.actions_mask
            self.steps += 1
            (node_feats, edge_index, it_index) = self.state
            data = Data(
                x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.int)
                .transpose(0, 1)
                .contiguous(),
            ).to(device)

            with torch.no_grad():
                action, action_log_prob, entropy, value = model(
                    data, self.actions_mask.to(device)
                )
                action = action.item()
                action_log_prob = action_log_prob.item()
                value = value.item()
            (
                total_speedup,
                new_node_feats,
                new_edge_index,
                legality,
                actions_mask,
                done,
            ) = apply_flattened_action(
                self.tiramisu_api,
                action,
                np.copy(node_feats),
                np.copy(edge_index),
                it_index,
                worker_id=str(self.worker_id),
            )
            self.actions_mask = torch.tensor(actions_mask)

            reward = self.reward_process(action, legality, total_speedup)

            trajectory.append(
                (
                    (np.copy(node_feats), np.copy(edge_index)),
                    action,
                    reward,
                    value,
                    action_log_prob,
                    entropy,
                    prev_actions_mask,
                )
            )
            self.state = (new_node_feats, new_edge_index, it_index)

            log_trajectory += (
                f"\nStep : {self.steps}"
                + f"\nAction ID : {action}"
                + f"\nLegality : {legality}"
                + f"\nActions Sequence So far : {self.tiramisu_api.scheduler_service.schedule_object.schedule_str}"
                + "\n"
            )

            if self.steps == 20 : 
                done = True

        else:
            schedule_object = self.tiramisu_api.scheduler_service.schedule_object

            tiramisu_program_dict = (
                self.tiramisu_api.get_current_tiramisu_program_dict()
            )
            self.dataset_worker.update_dataset.remote(
                self.current_program, tiramisu_program_dict
            )

        return {
            "trajectory": trajectory,
            "speedup": self.previous_speedup,
            "schedule_object": schedule_object,
            "log_trajectory": log_trajectory,
        }

    def reward_process(self, action, legality, total_speedup):
        switching_branch_penality = 0.9
        illegal_action_penality = 0.9
        max_speedup = np.inf
        log_base = 4

        if legality:
            if action != 55:
                # If the action is not Next
                instant_speedup = total_speedup / self.previous_speedup
                self.previous_speedup = total_speedup
            else:
                instant_speedup = switching_branch_penality
        else:
            instant_speedup = illegal_action_penality

        instant_speedup = np.clip(instant_speedup, 0, max_speedup)

        reward = math.log(instant_speedup, log_base)

        return reward
