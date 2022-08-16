from typing import Dict, List

from torch import nn

from Dataloader import Dataloader
from RDGNN_Config import RDGNN_Config

from PointConv import PointConv

class RDGNN():

    # Takes in config, and initializes.
    # All permanent state should be in config.
    def __init__(self, config: RDGNN_Config) -> None:
        self.config = config
        total_objects =10
        self.one_hot_encoding_embed_model = nn.Sequential(
                    nn.Linear(total_objects, self.config.one_hot_encoding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.config.one_hot_encoding_dim, self.config.one_hot_encoding_dim)
                )

        self.point_embed_model = PointConv(normal_channel=False)

        self.graph_encoding_model = ()

        self.graph_decoder = ()

    
    #Trains model given a dataloader.
    def run_model(self, dataloader: Dataloader ) -> int:
        data = dataloader.get_next()

        timesteps = [0, -1]

        for object in data["objects"].keys():
            
            pointcloud_embedding = self.point_embed_model(data["point_clouds"][0][object])


    def update_weights(self) -> None:
        pass

    #Returns relations for a given sample
    def predict_relations(self, data :Dict, action :List) -> List:
        pass

    