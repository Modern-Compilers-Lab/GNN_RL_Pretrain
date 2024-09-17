import pickle
import random
import os
import numpy as np
from env_api.core.models.tiramisu_program import TiramisuProgram
from utils.dataset_actor.services.base_data_service import (
    BaseDataService,
)


class HybridDataService(BaseDataService):
    def __init__(
        self,
        dataset_path: str,
        cpps_path: str,
        path_to_save_dataset: str,
        shuffle: bool = False,
        seed: int = None,
        saving_frequency: int = 10000,
    ):
        super().__init__(
            dataset_path=dataset_path,
            path_to_save_dataset=path_to_save_dataset,
            shuffle=shuffle,
            seed=seed,
            saving_frequency=saving_frequency,
        )
        self.cpps_path = cpps_path
        self.cpps = {}

        print(
            f"reading dataset in Hybrid format: dataset pkl from {self.dataset_path} and cpps from {self.cpps_path}"
        )

        with open(self.dataset_path, "rb") as f:
            self.dataset = pickle.load(f)
            self.function_names = os.listdir(self.cpps_path)

        # Shuffle the dataset (can be used with random sampling turned off to get a random order)
        if self.shuffle:
            # Set the seed if specified (for reproducibility)
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.function_names)

        self.dataset_size = len(self.function_names)

    # Returns next function name, function data, and function cpps
    def get_next_function(self, random=False) -> TiramisuProgram:
        if random:
            function_name = np.random.choice(self.function_names)
        # Choose the next function sequentially
        else:
            function_name = self.function_names[
                self.current_function_index % self.dataset_size
            ]
            self.current_function_index += 1

        # print(
        #     f"Selected function with index: {self.current_function_index}, name: {function_name}"
        # )
        if(function_name in self.dataset):
            # If the function doesn't exit in the dataset it will be compiled automatically
            return function_name, self.dataset[function_name], None
        else : 
            return function_name, None, None
