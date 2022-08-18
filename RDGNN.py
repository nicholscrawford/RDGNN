from typing import Dict, List, OrderedDict

from torch import nn
from torch_geometric.data import Batch

from Dataloader import Dataloader
from RDGNN_Config import RDGNN_Config

from PointConv import PointConv
import torch
import numpy as np
from data_utils import create_graph
from GNNOptionalEdge import GNNModelOptionalEdge

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

        self.graph_encoding_model = GNNModelOptionalEdge(
                    self.config.node_emb_size, 
                    self.config.edge_emb_size,
                    node_output_size = self.config.node_emb_size,
                    relation_output_size = self.config.relation_output_size, 
                    predict_edge_output = True,
                    edge_output_size = self.config.edge_emb_size,
                    graph_output_emb_size=16, 
                    node_emb_size=self.config.node_emb_size, 
                    edge_emb_size=self.config.edge_emb_size,
                    message_output_hidden_layer_size=128,  
                    message_output_size=128, 
                    node_output_hidden_layer_size=64,
                    all_classifier = False,
                    predict_obj_masks=False,
                    predict_graph_output=False,
                    use_edge_embedding = False,
                    use_edge_input = False
        )

        self.graph_decoding_model = GNNModelOptionalEdge(
                    self.config.node_emb_size, 
                    self.config.edge_emb_size,
                    node_output_size = self.config.node_emb_size, 
                    relation_output_size = self.config.relation_output_size, 
                    predict_edge_output = True,
                    edge_output_size = self.config.edge_emb_size,
                    graph_output_emb_size=16, 
                    node_emb_size=self.config.node_emb_size, 
                    edge_emb_size=self.config.edge_emb_size,
                    message_output_hidden_layer_size=128,  
                    message_output_size=128, 
                    node_output_hidden_layer_size=64,
                    all_classifier = False,
                    predict_obj_masks=False,
                    predict_graph_output=False,
                    use_edge_embedding = False,
                    use_edge_input = True
                )

    
    #Trains model given a dataloader.
    def run_model(self, dataloader: Dataloader ) -> int:
        data, attrs = dataloader.get_next()
        device = torch.device("cpu")

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
            #Obj_ids = A[len(self.config.environment_object_names):num_objects-len(self.config.environment_object_names)+1]
            Obj_ids = A[len(self.config.environment_object_names):]
            #Shuffle object ids for each training step
            np.random.shuffle(Obj_ids)
            Obj_ids = Obj_ids[:num_objects-len(self.config.environment_object_names)]
            
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
                one_hot_encoding[i][list(ids.values())[i]] = 1

            one_hot_encoding_tensor = torch.Tensor(np.array(one_hot_encoding))
            one_hot_encoding_embedding = self.one_hot_encoding_embed_model(one_hot_encoding_tensor)
            #print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
            pc_and_ohe_embedding = torch.cat([pointcloud_embedding, one_hot_encoding_embedding], dim = 1)

            #Create action
            #Not sure about required size? How should actions be encoded exactly
            if(attrs['behavior_params']['']['type'] == "PushObject"):
                action = np.zeros(8)
                target_object = attrs['behavior_params']['']['target_object']
                target_id = attrs['segmentation_labels'].index(target_object)#Assumption about ordering is here too
                action[list(ids.values())[target_id]] = 1
                for i in range(3):
                    action = np.append(action, attrs['behavior_params']['']['target_object_pose'][i] - attrs['behavior_params']['']['init_object_pose'][i])
                action = np.array([action for _ in range(len(data['objects']))])
            else:
                print(f"ERROR: Behavior to action not implemented for behavior type: {attrs['behavior_params']['']['type']}")
                exit   
            action = torch.FloatTensor(action)
            action.to(device)

            #Create Graph Size?
            object_graph = create_graph(len(data['objects']), pc_and_ohe_embedding, None, action)



            #Run model fr fr
            batch = Batch.from_data_list([object_graph]).to(device)
            #print(batch)
            outs = self.graph_encoding_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
            
            
            data_1_decoder = create_graph(len(data['objects']), outs['pred'], outs['pred_edge'], action)
            
            batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
            
            outs_decoder = self.graph_decoding_model(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
            
            

    def update_weights(self) -> None:
        pass

    #Returns relations for a given sample
    def predict_relations(self, data :Dict, action :List) -> List:
        pass

    