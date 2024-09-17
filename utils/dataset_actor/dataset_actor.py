import ray
import config.config as cfg
import env_api.core.models.tiramisu_program as tiramisu_program
from utils.dataset_actor.services.hybrid_data_service import HybridDataService
from utils.dataset_actor.services.pickle_data_service import PickleDataService

# Frequency at which the dataset is saved to disk
SAVING_FREQUENCY = 10000


@ray.remote
class DatasetActor:
    """
    DatasetActor is a class that is used to read the dataset and update it.
    It is used to read the dataset from disk and update it with the new functions.
    It is also used to save the dataset to disk.

    """

    def __init__(
        self,
        config: cfg.DatasetConfig,
    ):
        if config.dataset_format == cfg.DatasetFormat.PICKLE:
            self.dataset_service = PickleDataService(
                config.dataset_path, config.cpps_path, config.save_path, config.shuffle, config.seed, config.saving_frequency)
        elif config.dataset_format == cfg.DatasetFormat.HYBRID:
            self.dataset_service = HybridDataService(
                config.dataset_path, config.cpps_path, config.save_path, config.shuffle, config.seed, config.saving_frequency)
        else:
            raise ValueError("Unknown dataset format")

    def get_next_function(self, random=False) -> tiramisu_program.TiramisuProgram:
        return self.dataset_service.get_next_function(random)

    # Update the dataset with the new function
    def update_dataset(self, function_name: str, function_dict: dict) -> bool:
        return self.dataset_service.update_dataset(function_name, function_dict)

    # Get dataset size
    def get_dataset_size(self) -> int:
        return self.dataset_service.dataset_size