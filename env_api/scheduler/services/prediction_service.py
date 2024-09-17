from env_api.core.services.compiling_service import CompilingService
from env_api.scheduler.models.schedule import Schedule
from env_api.utils.exceptions import ExecutingFunctionException
import subprocess

INIT_TIMEOUT = 5 * 5 * 60 + 4
SLOWDOWN_TIMEOUT = 20


class PredictionService:
    def get_initial_time(self, schedule_object: Schedule):
        if "initial_execution" in schedule_object.prog.execution_times:
            # Original execution time of the program already exists in the dataset so we read the value directly
            initial_execution = schedule_object.prog.execution_times[
                "initial_execution"
            ]
        else:
            try:
                # We need to run the program to get the value
                initial_execution = CompilingService.execute_code(
                    tiramisu_program=schedule_object.prog,
                    optims_list=[],
                    timeout=INIT_TIMEOUT,
                )
                if initial_execution:
                    schedule_object.prog.execution_times[
                        "initial_execution"
                    ] = initial_execution
                else:
                    raise ExecutingFunctionException
            except subprocess.TimeoutExpired as e:
                schedule_object.prog.execution_times["initial_execution"] = None
                return None
            except ExecutingFunctionException as e :
                return None

        return initial_execution

    def get_real_speedup(self, schedule_object: Schedule):
        initial_execution = self.get_initial_time(schedule_object)

        if schedule_object.schedule_str in schedule_object.prog.execution_times:
            schedule_execution = schedule_object.prog.execution_times[
                schedule_object.schedule_str
            ]

        else:
            try:
                # We need to run the program to get the value
                schedule_execution = CompilingService.execute_code(
                    tiramisu_program=schedule_object.prog,
                    optims_list=schedule_object.schedule_list,
                    timeout=((initial_execution / 1000) * SLOWDOWN_TIMEOUT) * 5 + 4
                )
                if schedule_execution:
                    schedule_object.prog.execution_times[
                        schedule_object.schedule_str
                    ] = schedule_execution
                else:
                    raise ExecutingFunctionException
                
            except subprocess.TimeoutExpired as e:
                schedule_execution = initial_execution * SLOWDOWN_TIMEOUT
            
        return initial_execution / schedule_execution
