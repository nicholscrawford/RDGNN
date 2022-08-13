from typing import Dict, List

from torch import nn


from RDGNN_Config import RDGNN_Config

class RDGNN():

    # Takes in config, and initializes.
    # All permanent state should be in config.
    def __init__(self, config: RDGNN_Config) -> None:
        self.config = config

        self.one_hot_encoding_embed = nn.Sequential(
                    nn.Linear(total_objects, self.config.one_hot_encoding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.config.one_hot_encoding_dim, self.config.one_hot_encoding_dim)
                )

    
    #Trains model given a dataloader.
    def train(self, dataloader) -> None:
        pass


    #Returns relations for a given sample
    def predict_relations(self, data :Dict, action :List) -> List:
        pass

    