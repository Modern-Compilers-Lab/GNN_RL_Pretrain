import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

# Enum for the dataset format

class DatasetFormat:
    # Both the CPP code and the data of the functions are loaded from PICKLE files
    PICKLE = "PICKLE"
    # We look for informations in the pickle files, if something is missing we get it from cpp files in a dynamic way
    HYBRID = "HYBRID"

    @staticmethod
    def from_string(s):
        if s == "PICKLE":
            return DatasetFormat.PICKLE
        elif s == "HYBRID":
            return DatasetFormat.HYBRID
        else:
            raise ValueError("Unknown dataset format")

@dataclass
class TiramisuConfig:
    tiramisu_path: str = ""
    workspace: str = "./workspace/"
    experiment_dir: str = ""

@dataclass
class DatasetConfig:
    dataset_format: DatasetFormat = DatasetFormat.HYBRID
    cpps_path: str = ""
    dataset_path: str = ""
    save_path: str = ""
    shuffle: bool = False
    seed: int = None
    saving_frequency: int = 10000
    is_benchmark: bool =False

    def __init__(self, dataset_config_dict: Dict):
        self.dataset_format = DatasetFormat.from_string(
            dataset_config_dict["dataset_format"])
        self.cpps_path = dataset_config_dict["cpps_path"]
        self.dataset_path = dataset_config_dict["dataset_path"]
        self.save_path = dataset_config_dict["save_path"]
        self.models_save_path = dataset_config_dict["models_save_path"]
        self.results_save_path = dataset_config_dict["results_save_path"]
        self.evaluation_save_path = dataset_config_dict["evaluation_save_path"]
        self.shuffle = dataset_config_dict["shuffle"]
        self.seed = dataset_config_dict["seed"]
        self.saving_frequency = dataset_config_dict["saving_frequency"]
        self.is_benchmark = dataset_config_dict["is_benchmark"]

        if dataset_config_dict['is_benchmark']:
            self.dataset_path = dataset_config_dict["benchmark_dataset_path"] if dataset_config_dict[
                "benchmark_dataset_path"] else self.dataset_path
            self.cpps_path = dataset_config_dict["benchmark_cpp_files"] if dataset_config_dict[
                "benchmark_cpp_files"] else self.cpps_path

@dataclass
class Experiment:
    legality_speedup: float = 1.0
    beam_search_order: bool = False
    max_time_in_minutes: float = 1.0
    max_slowdown: float = 80
    DYNAMIC_RUNS: int = 0
    MAX_RUNS: int = 5
    NB_EXEC: int = 3

@dataclass
class EnvVars:
    CXX: str = ""
    TIRAMISU_ROOT: str = ""
    CONDA_ENV: str = ""
    LD_LIBRARY_PATH: str = ""

@dataclass
class AutoSchedulerConfig:

    tiramisu: TiramisuConfig
    dataset: DatasetConfig
    experiment: Experiment
    env_vars: EnvVars

    def __post_init__(self):
        if isinstance(self.tiramisu, dict):
            self.tiramisu = TiramisuConfig(**self.tiramisu)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(self.dataset)
        if isinstance(self.experiment, dict):
            self.experiment = Experiment(**self.experiment)
        if isinstance(self.env_vars, dict):
            self.env_vars = EnvVars(**self.env_vars)

def read_yaml_file(path):
    with open(path) as yaml_file:
        return yaml_file.read()

def parse_yaml_file(yaml_string: str) -> Dict[Any, Any]:
    return yaml.safe_load(yaml_string)

def dict_to_config(parsed_yaml: Dict[Any, Any]) -> AutoSchedulerConfig:
    tiramisu = TiramisuConfig(**parsed_yaml["tiramisu"])
    dataset = DatasetConfig(parsed_yaml["dataset"])
    experiment = Experiment(**parsed_yaml["experiment"])
    env_vars = EnvVars(**parsed_yaml["env_vars"])
    return AutoSchedulerConfig(tiramisu, dataset,  experiment, env_vars)

class Config(object):
    config = None

    @classmethod
    def init(self, config_yaml="./config/config.yaml"):
        parsed_yaml_dict = parse_yaml_file(read_yaml_file(config_yaml))
        Config.config = dict_to_config(parsed_yaml_dict)
