from typing import Dict, List

from torch import nn

from Dataloader import Dataloader
from RDGNN_Config import RDGNN_Config

from PointConv import PointConv
import torch
import numpy as np

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
        pointcloud_embedding_list = []
        for timestep in timesteps:
            pointclouds = data['point_clouds'][timestep]
            reshaped_pointclouds = []
            for object in data["objects"].keys():
                objectpc = pointclouds[object].T
                reshaped_pointclouds.append(objectpc)

                #embed model takes [3, 3, 128] for 3 objects, three data, and 128 of each
                #num objects should then be part of the confg for now.

            pointclouds_tensor = torch.FloatTensor(np.array(reshaped_pointclouds))
            pointcloud_embedding = self.point_embed_model(pointclouds_tensor)
            pointcloud_embedding_list.append(pointcloud_embedding)


    def update_weights(self) -> None:
        pass

    #Returns relations for a given sample
    def predict_relations(self, data :Dict, action :List) -> List:
        pass

    