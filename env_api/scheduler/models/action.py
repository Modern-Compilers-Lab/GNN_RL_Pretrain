from typing import List


class Action:
    def __init__(
        self,
        params: list,
        name: str,
        comps: list = [],
        env_id: int = None,
        worker_id="",
    ):
        self.params = params
        self.name = name
        # List of computations concerned by this action
        self.comps = comps
        # The ID of the action inside the RL system, this ID helps in masking actions and matching RL with env_api
        self.env_id = env_id
        # In distributed training we want to know which worker is applying this action in order to distinguish the compilation
        # of the same function by different workers
        self.worker_id = worker_id

        self.legality_code_str = ""
        self.execution_code_str = ""

        self.comps_schedule = {}
    
    def set_comps(self, comps):
        self.comps = comps

    def apply_on_branches(self, branches ):
        for comp in self.comps:
            for branch in branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                    # Update the actions mask 
                    branch.update_actions_mask(action=self)


class AffineAction(Action):
    def __init__(
        self,
        params: list,
        name: str,
        comps: list = [],
        env_id: int = None,
        worker_id="",
    ):
        super().__init__(params, name, comps, env_id, worker_id)

    def set_comps(self, comps):
        super().set_comps(comps)

    def apply_on_branches(self, branches):
        super().apply_on_branches(branches)


class Reversal(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Reversal", env_id=env_id, worker_id=worker_id)

    def set_comps(self, comps):
        super().set_comps(comps)
        loop_level = self.params[0]
        optim_str = ""
        for comp in self.comps:
            self.comps_schedule[comp] = f"R(L{loop_level})"
            optim_str += f"\n\t{comp}.loop_reversal({loop_level});"

        self.legality_code_str = self.execution_code_str = optim_str


class Interchange(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Interchange", env_id=env_id, worker_id=worker_id)

    def set_comps(self, comps):
        super().set_comps(comps)
        loop_level1 , loop_level2 = self.params
        optim_str = ""
        for comp in self.comps:
            self.comps_schedule[comp] = f"I(L{loop_level1},L{loop_level2})"
            optim_str += f"\n\t{comp}.interchange({loop_level1}, {loop_level2});"

        self.legality_code_str = self.execution_code_str = optim_str


class Skewing(AffineAction):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Skewing", env_id=env_id, worker_id=worker_id)

    def set_comps(self, comps):
        super().set_comps(comps)

    def set_factors(self, factors):
        self.params.extend(factors)

        loop_level1, loop_level2, factor1, factor2 = self.params
        optim_str = ""
        for comp in self.comps:
            self.comps_schedule[comp] = f"S(L{loop_level1},L{loop_level2},{factor1},{factor2})"
            optim_str += f"\n\t{comp}.skew({loop_level1}, {loop_level2}, {factor1}, {factor2});"
        
        self.legality_code_str = self.execution_code_str = optim_str

class Parallelization(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(
            params, name="Parallelization", env_id=env_id, worker_id=worker_id
        )
    def set_comps(self, comps):
        super().set_comps(comps)

        loop_level = self.params[0]

        for comp in self.comps:
            self.comps_schedule[comp] = f"P(L{loop_level})"
        
        comp = self.comps[0]

        self.execution_code_str = f"\n\t{comp}.tag_parallel_level({loop_level});"
        
        self.legality_code_str = (
        f"""\n\tprepare_schedules_for_legality_checks(true);
        is_legal &= loop_parallelization_is_legal({loop_level}, {{&{",&".join(self.comps)}}});
        {self.execution_code_str}
        """
        )
        


class Unrolling(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Unrolling", env_id=env_id, worker_id=worker_id)

    def set_comps(self, comps):
        super().set_comps(comps)

        loop_level , factor = self.params
        optim_str = ""
        for comp in self.comps:
            self.comps_schedule[comp] = f"U(L{loop_level},{factor})"
            optim_str += f"\n\t{comp}.unroll({loop_level},{factor});"

        self.execution_code_str = optim_str

        self.legality_code_str = (
        f"""\n\tprepare_schedules_for_legality_checks(true);
        is_legal &= loop_unrolling_is_legal({loop_level}, {{&{",&".join(self.comps)}}});
        {self.execution_code_str}
        """
        )

    def set_params(self, additional_loops):
        loop_level , factor = self.params
        loop_level += additional_loops
        optim_str = ""
        for comp in self.comps:
            self.comps_schedule[comp] = f"U(L{loop_level},{factor})"
            optim_str += f"\n\t{comp}.unroll({loop_level},{factor});"

        self.execution_code_str = optim_str

        self.legality_code_str = (
        f"""\n\tprepare_schedules_for_legality_checks(true);
        is_legal &= loop_unrolling_is_legal({loop_level}, {{&{",&".join(self.comps)}}});
        {self.execution_code_str}
        """
        )       

    def apply_on_branches(self, branches, current_branch : int ):
        branches[current_branch].update_actions_mask(action=self)



class Tiling(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Tiling", env_id=env_id, worker_id=worker_id)

    def set_comps(self, comps):
        super().set_comps(comps)

        size = len(self.params)//2
        loop_args = "".join([f"L{c}," for c in self.params[:size]])
        factor_args = ",".join([f"{c}" for c in self.params[size:]])
        optim_str = ""
        for comp in self.comps:
            self.comps_schedule[comp] = f"T{size}({loop_args}{factor_args})"
            optim_str += f"\n\t{comp}.tile({','.join([str(p) for p in self.params])});"

        self.execution_code_str = self.legality_code_str = optim_str

    def apply_on_branches(self, branches, schedule_list: List[Action]):
        tiling_depth = len(self.params)//2
        for comp in self.comps:
            for branch in branches : 
                # Check for the branches that needs to be updated
                if (comp in branch.comps):
                    # Update the actions mask 
                    branch.update_actions_mask(action=self)
                    branch.additional_loops = tiling_depth

            for optim in schedule_list: 
                if isinstance(optim, Unrolling):
                    if comp in optim.comps:
                        optim.set_params(additional_loops=tiling_depth)

class Fusion(Action):
    def __init__(self, params: list, env_id: int = None, worker_id=""):
        super().__init__(params, name="Fusion", env_id=env_id, worker_id=worker_id)
