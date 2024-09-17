from abc import abstractmethod
import pickle
from typing import Tuple

from env_api.core.services.tiramisu_service import TiramisuService


class BaseDataService():
    def __init__(self, dataset_path: str, path_to_save_dataset: str, shuffle: bool = False, seed: int = None, saving_frequency: int = 10000) -> None:
        self.dataset_path = dataset_path
        self.path_to_save_dataset = path_to_save_dataset
        self.shuffle = shuffle
        self.seed = seed
        self.saving_frequency = saving_frequency

        self.dataset = {}
        self.function_names = []
        self.dataset_size = 0
        self.current_function_index = 0
        self.nbr_updates = 0
        self.dataset_name = dataset_path.split("/")[-1].split(".")[0]
        self.tiramisu_service = TiramisuService()

    @abstractmethod
    def get_next_function(self, random=False) -> Tuple:
        pass

    # Update the dataset with the new function
    def update_dataset(self, function_name: str, function_dict: dict) -> bool:
        """
        Update the dataset with the new function
        :param function_name: name of the function
        :param function_dict: dictionary containing the function schedules
        :return: True if the dataset was saved successfully
        """
        for key in function_dict.keys():
            self.dataset[function_name][key] = function_dict[key]

        self.nbr_updates += 1
        # print(f"# updates: {self.nbr_updates}")
        if self.nbr_updates % self.saving_frequency == 0:
            if self.nbr_updates % (2 * self.saving_frequency):
                return self.save_dataset_to_disk(version=2)
            else:
                return self.save_dataset_to_disk(version=1)
        return False

    # Save the dataset to disk
    def save_dataset_to_disk(self, version=1) -> bool:
        """
        Save the dataset to disk
        :param version: version of the dataset to save (1 or 2)
        :return: True if the dataset was saved successfully
        """
        print("[Start] Save the legality_annotations_dict to disk")

        updated_dataset_name = (
            f"{self.path_to_save_dataset}/{self.dataset_name}_updated_{version}"
        )
        with open(f"{updated_dataset_name}.pkl", "wb") as f:
            pickle.dump(self.dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("[Done] Save the legality_annotations_dict to disk")
        return True
