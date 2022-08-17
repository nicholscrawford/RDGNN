from typing import Dict
import os
import pickle
# Dataloader is responsible for 
# taking in a set of data points,
# transforming to correct format and saving if needed
# and storing data in memory, 
# 
# and returning samples/batches of samples when needed.

class Dataloader():
    
    def __init__(self, train_dir_list : str) -> None:
        self.files = []

        for train_dir in train_dir_list:
            files = sorted(os.listdir(train_dir))
            files = [(os.path.join(train_dir, f)) for f in files if "demo" in f]
            for file in files:
                self.files.append(file)

        self.current_file_idx = 0
        #Hold everything in memory? Load batches and then use them?
        #Temp solution is to just load from disk every time.
        #for file in self.files:


    def get_next(self) -> Dict:
        with open(self.files[self.current_file_idx], 'rb') as file:
            data, attributes = pickle.load(file)

        self.current_file_idx += 1
        return data, attributes
