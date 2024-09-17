from env_api.core.models.tiramisu_program import TiramisuProgram
from env_api.scheduler.models.schedule import Schedule


class Branch(Schedule):
    def __init__(self, program: TiramisuProgram):
        super().__init__(program)
        # For a branch we don't need to recalculate the common iterators because all of them are common.
        self.common_it = list(program.annotations["iterators"].keys())
