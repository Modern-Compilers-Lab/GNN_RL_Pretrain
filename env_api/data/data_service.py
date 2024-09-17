from config.config import Config
import pickle
from datetime import date


class DataSetService:
    def __init__(self,
                 cpps_dataset_path=None,
                 schedules_dataset_path=None):
        self.cpps_dataset_path = cpps_dataset_path
        self.schedules_dataset_path = schedules_dataset_path
        self.schedules_dataset = None
        self.cpps_dataset = None
        try :
            with open(cpps_dataset_path, "rb") as file:
                    self.cpps_dataset = pickle.load(file)
        except FileNotFoundError:
            print("[Error] : Offline dataset path is not valid => Reading from cpp files on disk")

        if (schedules_dataset_path != None):
            try :
                with open(schedules_dataset_path, "rb") as file:
                    self.schedules_dataset = pickle.load(file)
            except FileNotFoundError:
                print("[Error] : Offline dataset path is not valid => Reading from cpp files on disk")

    def in_schedule_dataset(self, func_name):
        exist_offline = False
        # Check if the file exists in the offline dataset
        if (self.schedules_dataset):
            # Checking the function name
            exist_offline = func_name in self.schedules_dataset
        return exist_offline

    def get_offline_prog_data(self, name: str):
        return self.schedules_dataset[name]
    
    def get_prog_code(self , name:str):
        return self.cpps_dataset[name]

    def store_offline_dataset(self,suffix:str = ""):
        if(self.schedules_dataset):
            with open(self.schedules_dataset_path[:-4] + date.today().__str__() +suffix+".pkl", "wb") as file:
                pickle.dump(self.schedules_dataset,file)