from typing import Dict, List, OrderedDict

from torch import nn
from torch_geometric.data import Batch

from Dataloader import Dataloader
from RDGNN_Config import RDGNN_Config

from PointConv import PointConv
import torch
import numpy as np
from data_utils import create_graph

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
        data, attrs = dataloader.get_next()
        device = torch.cuda.Device('cuda')

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

            #Create object ids
            num_objects = len(data['objects'].keys())
            A = np.arange(self.config.max_objects)
            Env_ids = A[:len(self.config.environment_object_names)] #Env ids are unique to each env object, starting at zero.
            Obj_ids = A[len(self.config.environment_object_names):num_objects-len(self.config.environment_object_names)]
            #Shuffle object ids for each training step
            np.random.shuffle(Obj_ids)
            
            ids = OrderedDict()
            e = 0
            o = 0
            for object_name in data['objects'].keys():
                if object_name in self.config.environment_object_names:
                    ids[object_name] = Env_ids[e]
                    e += 1
                else:
                    ids[object_name] = Obj_ids[o]
                    o += 1



            # Create One Hot Encodings
            # Theres some implicit object ordering going on, with the point clouds and one hot encoded ids.
            one_hot_encoding = np.zeros((len(data['objects'].keys()), self.config.max_objects))
            for i in range(num_objects):
                one_hot_encoding[i][List(ids.values())[i]] = 1

            one_hot_encoding_tensor = torch.Tensor(np.array(one_hot_encoding))
            one_hot_encoding_embedding = self.one_hot_encoding_embed_model(one_hot_encoding_tensor)
            #print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
            pc_and_ohe_embedding = torch.cat([pointcloud_embedding, one_hot_encoding_embedding], dim = 1)

            #Create action
            #Not sure about required size? How should actions be encoded exactly
            if(attrs['behavior_params']['']['type'] == "PushObject"):
                action = np.zeros(len(data[object].keys()), 8)
                target_object = attrs['behavior_params']['']['target_object']
                target_id = attrs['segmentation_labels'].index(target_object)#Assumption about ordering is here too
                action[List(ids.values())[target_id]] = 1
                for i in range(3):
                    action.append(attrs['behavior_params']['']['target_object_pose'][i] - attrs['behavior_params']['']['init_object_pose'][i])
            else:
                print(f"ERROR: Behavior to action not implemented for behavior type: {attrs['behavior_params']['']['type']}")
                exit   


            #Create Graph
            data = create_graph(self, len(data['objects']), pc_and_ohe_embedding, None, action)



            #Run model fr fr
            batch = Batch.from_data_list([data]).to(device)
            #print(batch)
            outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
            
            
            data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action)
            
            batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
            
            outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
            
            

    def update_weights(self) -> None:
        pass

    #Returns relations for a given sample
    def predict_relations(self, data :Dict, action :List) -> List:
        pass

    