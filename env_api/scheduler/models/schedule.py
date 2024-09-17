import numpy as np , copy
from config.config import Config
from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.scheduler.models.action import *

class Schedule:
    def __init__(self, program: TiramisuProgram):
        self.schedule_str = ""
        self.prog = program
        # List of computations of the program
        self.comps = self.prog.comps
        # Iterators dictionnary
        self.it_dict = {}
        # List of branches of the program tree
        self.branches = []
        # List of common iterators
        self.common_it = []
        # self.schedule_list is an array that contains a list of optimizations that has been applied on the program
        # This list has objects of type `OptimizationCommand`
        self.schedule_list = []
        # Additional loops when Tiling is applied
        self.additional_loops = 0

        self.actions_mask = None


        if((type(self).__name__) == "Schedule"):
            self.__calculate_common_it()
            self.__set_action_mask()
            self.__form_iterators_dict()
            self.__form_branches()
        else : 
            self.__set_action_mask()
            self.__form_iterators_dict()


    def __calculate_common_it(self):
        if len(self.comps) != 1:  # Multi-computation program
            # comps_it is a list of lists of iterators of computations
            comps_it = []
            for comp in self.comps:
                comps_it.append(
                    self.prog.annotations["computations"][comp]["iterators"]
                )
            self.common_it = comps_it[0]
            for comp_it in comps_it[1:]:
                self.common_it = [it for it in comp_it if it in self.common_it]
        else:  # A single comp program
            self.common_it = self.prog.annotations["computations"][self.comps[0]][
                "iterators"
            ]
    
    def __set_action_mask(self):
        self.actions_mask = np.zeros(56)

    def __form_iterators_dict(self):
        for comp in self.comps:
            comp_it_dict = {}
            iterators = list(self.prog.annotations["computations"][comp]["iterators"])
            for i in range(len(iterators)):
                comp_it_dict[i] = {}
                comp_it_dict[i]['iterator'] = iterators[i]
                comp_it_dict[i]['lower_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['lower_bound']
                comp_it_dict[i]['upper_bound'] = self.prog.annotations['iterators'][
                    iterators[i]]['upper_bound']
            self.it_dict[comp] = comp_it_dict
            
    def __form_branches(self):
        branches = []
        iterators = copy.deepcopy(self.prog.annotations["iterators"])
        computations = copy.deepcopy(self.prog.annotations["computations"])
        it = {}
        for computation in computations:
            iterators = copy.deepcopy(self.prog.annotations["computations"][computation]["iterators"])
            if iterators[-1] in it :
                it[iterators[-1]]["comps"].append(computation)
            else :
                it[iterators[-1]] = {
                    "comps" : [computation],
                    "iterators" : iterators
                }
        
        for iterator in it :
            branches.append({
                "comps" : it[iterator]["comps"],
                "iterators" : it[iterator]["iterators"],
                "annotations": {}
            })
                
        for branch in branches :
            branch_annotations = {
                "computations" : {},
                "iterators": {}
            }
            for comp in branch["comps"]:
                branch_annotations["computations"][comp] = copy.deepcopy(self.prog.annotations["computations"][comp])
            # extract the branch specific iterators annotations
            for iterator in branch["iterators"]:
                branch_annotations["iterators"][iterator] = copy.deepcopy(self.prog.annotations["iterators"][iterator])
                if (self.prog.annotations["iterators"][iterator]["parent_iterator"]):
                    # Making sure that the parent node has the actual node as the only child
                    # It may happen that the parent node has many children but in a branch it is only allowed
                    # to have a single child to form a straight-forward branch from top to bottom
                    parent = (branch_annotations["iterators"][iterator]["parent_iterator"])
                    branch_annotations["iterators"][parent]["child_iterators"] = copy.deepcopy([iterator])
                    branch_annotations["iterators"][parent]["computations_list"] = []
            branch["annotations"] = copy.deepcopy(branch_annotations)

        self.branches = branches


    def build_sched_string(self) -> str:
        # Prepare a dictionary of computations name to fill it with each action applied on every comp
        comps = {}
        # Map the schedules applied one by one
        for schedule in self.schedule_list : 
            # schedule has comps_schedule which includes the comps that was invloved in the optimisation
            for key in schedule.comps_schedule.keys():
                # Add the data from that schedule to the global comps dictionnary
                if(not key in comps or not comps[key]):
                    comps[key] = ""
                comps[key] += schedule.comps_schedule[key]
        # Prepare the string and form it from the comps dictionary
        schedule_string = ""
        for key in comps.keys():
            schedule_string+= "{"+key+"}:"+comps[key]
        return schedule_string

    
    def update_actions_mask(self, action : Action,applied : bool = True):
        # Whether an action is legal or not we should mask it to not use it again
        self.actions_mask[action.env_id] = 1

        if applied and Config.config.experiment.beam_search_order:
            self.apply_beam_search_conditions(action=action)
    
    def apply_beam_search_conditions(self, action : Action):
        # The order of actions in beam search :
        # Fusion, [Interchange, reversal, skewing], parallelization, tiling, unrolling
        if (isinstance(action,Skewing)):
            self.actions_mask[0:12] = 1
            self.actions_mask[14:50] = 1


        elif (isinstance(action,Parallelization)):
            self.actions_mask[0:12] = 1

        elif (isinstance(action,Tiling)) : 
            self.actions_mask[0:50] = 1

        elif (isinstance(action,Unrolling)):
            self.actions_mask[0:55] = 1
