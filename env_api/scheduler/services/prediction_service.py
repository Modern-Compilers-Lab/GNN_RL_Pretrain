from env_api.core.services.compiling_service import CompilingService
from env_api.scheduler.models.schedule import Schedule
from env_api.utils.exceptions import ExecutingFunctionException
import subprocess
from config.config import Config

INIT_TIMEOUT = (5 * 5 * 60 + 4 )/10# 5 * 5 * 60 + 4
SLOWDOWN_TIMEOUT = 20


class PredictionService:
    def get_initial_time(self, schedule_object: Schedule):
        initial_execution = schedule_object.prog.get_execution_time(
            "initial_execution", Config.config.machine
        )
        
        if initial_execution is None:
            try:
                # We need to run the program to get the value
                initial_execution = CompilingService.execute_code(
                    tiramisu_program=schedule_object.prog,
                    optims_list=[],
                    timeout=INIT_TIMEOUT,
                )
                if initial_execution:
                    schedule_object.prog.execution_times[Config.config.machine][
                        "initial_execution"
                    ] = initial_execution
                else:
                    raise ExecutingFunctionException
            except subprocess.TimeoutExpired :
                return None
            except ExecutingFunctionException :
                return None

        return initial_execution

    def get_real_speedup(self, schedule_object: Schedule):
        initial_execution = self.get_initial_time(schedule_object)
        num_hits = 0
        schedule_execution = schedule_object.prog.get_execution_time(
            schedule_object.schedule_str, Config.config.machine
        )
        if schedule_execution is None:
            try:
                # We need to run the program to get the value
                schedule_execution = CompilingService.execute_code(
                    tiramisu_program=schedule_object.prog,
                    optims_list=schedule_object.schedule_list,
                    timeout=((initial_execution / 1000) * SLOWDOWN_TIMEOUT) * 5 + 4
                )
                if schedule_execution:
                    schedule_object.prog.execution_times[Config.config.machine][
                        schedule_object.schedule_str
                    ] = schedule_execution
                else:
                    raise ExecutingFunctionException
                
            except subprocess.TimeoutExpired:
                schedule_execution = initial_execution * SLOWDOWN_TIMEOUT
        else:
            num_hits = num_hits + 1
            
        return initial_execution / schedule_execution, num_hits

