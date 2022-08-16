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

        self.one_hot_encoding_embed_model = nn.Sequential(
                    nn.Linear(self.config.max_objects, self.config.one_hot_encoding_dim),
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
        pc_and_ohe_embedding_list = []
        for timestep in timesteps:
            
            # Create Point Cloud Embeddings
            pointclouds = data['point_clouds'][timestep]
            reshaped_pointclouds = []
            for object in data["objects"].keys():
                objectpc = pointclouds[object].T
                reshaped_pointclouds.append(objectpc)
                #embed model takes [x, 3, 128] for x objects, three data, and 128 of each
            pointclouds_tensor = torch.FloatTensor(np.array(reshaped_pointclouds))
            pointcloud_embedding = self.point_embed_model(pointclouds_tensor)

            # Create One Hot Encodings
            ##TODO: Static encoding for environment
            A = np.arange(self.config.max_objects)
            np.random.shuffle(A)
            select_obj_num_range = A[:len(data['objects'].keys())]
            one_hot_encoding = np.zeros((len(data['objects'].keys()), self.config.max_objects))
            for i in range(len(select_obj_num_range)):
                one_hot_encoding[i][select_obj_num_range[i]] = 1

            one_hot_encoding_tensor = torch.Tensor(np.array(one_hot_encoding))
            one_hot_encoding_embedding = self.one_hot_encoding_embed_model(one_hot_encoding_tensor)
            #print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
            pc_and_ohe_embedding = torch.cat([pointcloud_embedding, one_hot_encoding_embedding], dim = 1)
            pc_and_ohe_embedding_list.append(pc_and_ohe_embedding)

    def update_weights(self) -> None:
        pass

    #Returns relations for a given sample
    def predict_relations(self, data :Dict, action :List) -> List:
        pass

    