from typing import List
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.scheduler.models.branch import Branch
from env_api.scheduler.services.legality_service import LegalityService
from env_api.scheduler.services.prediction_service import PredictionService
from env_api.utils.exceptions import ExecutingFunctionException
from ..models.schedule import Schedule
from ..models.action import *
from config.config import Config



class SchedulerService:
    def __init__(self):
        # The Schedule object contains all the informations of a program : annotatons , tree representation ...
        self.schedule_object: Schedule = None
        # The branches generated from the program tree 
        self.branches : List[Branch] = []
        self.current_branch = 0
        # The prediction service is an object that has a value estimator `get_predicted_speedup(schedule)` of the speedup that a schedule will have
        # This estimator is a recursive model that needs the schedule representation to give speedups
        self.prediction_service = PredictionService()
        # A schedules-legality service
        self.legality_service = LegalityService()
        

    def set_schedule(self, schedule_object: Schedule):
        """
        The `set_schedule` function is called first in `tiramisu_api` to initialize the fields when a new program is fetched from the dataset.
        input :
            - schedule_object : contains all the inforamtions on a program and the schedule
        output :
            - a tuple of vectors that represents the main program and the current branch , in addition to their respective actions mask
        """
        self.schedule_object = schedule_object
        # We create the branches of the program
        self.create_branches()
        # Init the index to the 1st branch
        self.current_branch = 0
        return self.branches[self.current_branch].actions_mask
                
  

    def create_branches(self):
        # Make sure to clear the branches of the previous function if there are ones
        self.branches.clear()
        for branch in self.schedule_object.branches : 
            # Create a mock-up of a program from the data of a branch
            program_data = {
                "program_annotation" : branch["annotations"],
                "schedules_legality" : {},
                "schedules_solver" : {}
            }
            # The Branch is an inherited class from Schedule, it has all its characteristics
            new_branch = Branch(TiramisuProgram.from_dict(self.schedule_object.prog.name,
                                                          data=program_data,
                                                          original_str=""))
            # The branch needs the original cpp code of the main function to calculate legality of schedules
            new_branch.prog.load_code_lines(self.schedule_object.prog.original_str)
            self.branches.append(new_branch)
            
    def next_branch(self):
        # Switch to the next branch to optimize it 
        self.current_branch += 1
        if (self.current_branch == len(self.branches)):
            # This matks the finish of exploring the branches
            return None
        # Using the model to embed the program and the branch in a 180 sized vector each
        return self.branches[self.current_branch].actions_mask
                

    def apply_action(self, action: Action):
        """
        input :
            - an action that represents an optimization from the 7 types : Parallelization,Skewing,Interchange,Fusion,Reversal,Tiling,Unrolling
        output :
            - speedup : float , representation : tuple(tensor) , legality_check : bool
        """
        legality_check = self.legality_service.is_action_legal(schedule_object=self.schedule_object,
                                                               branches=self.branches,
                                                               current_branch=self.current_branch,
                                                               action=action)
        speedup = 1
        if legality_check:
            try : 
                if (Config.config.dataset.is_benchmark):
                    speedup = 1
                else :
                    speedup = self.prediction_service.get_real_speedup(schedule_object=self.schedule_object)

                if isinstance(action, Tiling):
                    action.apply_on_branches(self.branches, self.schedule_object.schedule_list)
                    
                elif isinstance(action, Unrolling):
                    action.apply_on_branches(self.branches, self.current_branch)
                
                else : 
                    action.apply_on_branches(self.branches)

            except ExecutingFunctionException as e :
                # If the execution went wrong remove it from the schedule list
                self.schedule_object.schedule_list.pop()
                # Rebuild the scedule string after removing the action 
                schdule_str = self.schedule_object.build_sched_string()
                # Storing the schedule string to use it later 
                self.schedule_object.schedule_str = schdule_str
                legality_check = False

        

        return speedup, legality_check, self.branches[self.current_branch].actions_mask
