import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool



class GNNModelOptionalEdge(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 edge_inp_size,
                 node_output_size, 
                 relation_output_size, 
                 max_objects = 8, 
                 graph_output_emb_size=16, 
                 node_emb_size=32, 
                 edge_emb_size=32,
                 message_output_hidden_layer_size=128,  
                 message_output_size=128, 
                 node_output_hidden_layer_size=64,
                 edge_output_size=16,
                 use_latent_action = True,
                 latent_action_dim = 128, 
                 all_classifier = False,
                 predict_obj_masks=False,
                 predict_graph_output=False,
                 use_edge_embedding=False,
                 predict_edge_output=False,
                 use_edge_input=False,
                 node_embedding = False,
                 use_shared_latent_embedding = False,
                 use_seperate_latent_embedding = False,
                 use_env_data = False):
        # define the relation_output_size by hand for all baselines. 
        self.relation_output_size = relation_output_size
        # Make sure all the planning stuff keeps the same for all our comparison approaches. 
        super(GNNModelOptionalEdge, self).__init__(aggr='mean')
        # all edge output will be classifier
        self.all_classifier = all_classifier

        self.node_inp_size = in_channels
        # Predict if an object moved or not
        self._predict_obj_masks = predict_obj_masks
        # predict any graph level output
        self._predict_graph_output = predict_graph_output

        self.latent_action_dim = latent_action_dim
        self.use_latent_action = use_latent_action

        self.use_seperate_latent_embedding = use_seperate_latent_embedding
        
        
        self.use_one_hot_embedding = True
        if self.use_one_hot_embedding: 
            self.one_hot_encoding_dim = 128

        if  use_env_data:
            max_objects += 1
        
        total_objects = max_objects
        print('max-objects', max_objects)
        action_dim = total_objects + 3
        if use_shared_latent_embedding:
            action_dim = action_dim + 1
        if self.use_latent_action:
            self._in_channels = self.latent_action_dim
            self.action_emb = nn.Sequential(
                nn.Linear(action_dim, self.latent_action_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_action_dim, self.latent_action_dim)
            )
        else:
            self._in_channels = action_dim

        if self.use_one_hot_embedding: 
            self.one_hot_encoding_embed = nn.Sequential(
                    nn.Linear(total_objects, self.one_hot_encoding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
                )

        
        

        self._use_edge_dynamics = True

        self.use_edge_input = use_edge_input
        if self.use_edge_input:
            self.use_one_hot_embedding = False
        
        if use_edge_input == False:
            edge_inp_size = 0
            use_edge_embedding = False
            self._use_edge_dynamics = False
        self._edge_inp_size = edge_inp_size

        self._node_emb_size = node_emb_size
        self.node_embedding = node_embedding
        if self.node_embedding:
            self.node_emb = nn.Sequential(
                nn.Linear(in_channels, self._node_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(self._node_emb_size, self._node_emb_size)
            )
        if not self.node_embedding:
            if self.use_one_hot_embedding:
                self.node_inp_size += self.one_hot_encoding_dim
                self._node_emb_size = self.node_inp_size
                
            else:
                self._node_emb_size = self.node_inp_size

        self.edge_emb_size = edge_emb_size
        self._use_edge_embedding = use_edge_embedding
        self._test_edge_embedding = False
        if use_edge_embedding:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_inp_size, edge_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(edge_emb_size, edge_emb_size)
            )

        self._message_layer_size = message_output_hidden_layer_size
        self._message_output_size = message_output_size
        #print('node input size', self.node_inp_size)
        if self.node_embedding:
            message_inp_size = 2*self._node_emb_size + edge_emb_size if use_edge_embedding else \
                2 * self._node_emb_size + edge_inp_size
        else:
            message_inp_size = 2*self.node_inp_size + edge_emb_size if use_edge_embedding else \
                2 * self.node_inp_size + edge_inp_size
        # if use_edge_input == False:
        #     message_inp_size = 2 * self._node_emb_size
        self.message_info_mlp = nn.Sequential(
            nn.Linear(message_inp_size, self._message_layer_size),
            nn.ReLU(),
            # nn.Linear(self._message_layer_size, self._message_layer_size),
            # nn.ReLU(),
            nn.Linear(self._message_layer_size, self._message_output_size)
            )

        self._node_output_layer_size = node_output_hidden_layer_size
        self._per_node_output_size = node_output_size
        graph_output_emb_size = 0
        self._per_node_graph_output_size = graph_output_emb_size
        self.node_output_mlp = nn.Sequential(
            nn.Linear(self._node_emb_size + self._message_output_size, self._node_output_layer_size),
            nn.ReLU(),
            nn.Linear(self._node_output_layer_size, node_output_size + graph_output_emb_size)
        )

        action_dim = self._in_channels
        self.action_dim = action_dim
        self.dynamics =  nn.Sequential(
            nn.Linear(self._in_channels+action_dim, 128),  # larger value
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self._in_channels)
        )

        if self._use_edge_dynamics:
            self.edge_dynamics =  nn.Sequential(
                nn.Linear(self._edge_inp_size+action_dim, 128),  # larger value
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self._edge_inp_size)
            )

        
        if self.use_seperate_latent_embedding:
            self.graph_dynamics_0 = nn.Sequential(
                nn.Linear(node_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, node_output_size)
            )

            self.graph_edge_dynamics_0 = nn.Sequential(
                nn.Linear(edge_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, edge_output_size)
            )

            self.graph_dynamics_1 = nn.Sequential(
                nn.Linear(node_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, node_output_size)
            )

            self.graph_edge_dynamics_1 = nn.Sequential(
                nn.Linear(edge_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, edge_output_size)
            )
        else:
            self.graph_dynamics = nn.Sequential(
                nn.Linear(node_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, node_output_size)
            )

            self.graph_edge_dynamics = nn.Sequential(
                nn.Linear(edge_output_size+action_dim, 512),  # larger value
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, edge_output_size)
            )

        if self._predict_graph_output:
            self._graph_pred_mlp = nn.Sequential(
                nn.Linear(graph_output_emb_size, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
        
        self._should_predict_edge_output = predict_edge_output
        if predict_edge_output:
            self._edge_output_size = edge_output_size
            # TODO: Add edge attributes as well, should be easy
            if True:
                self._edge_output_mlp = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, edge_output_size)
                )
                self._edge_output_sigmoid = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.relation_output_size),
                    nn.Sigmoid()
                )
            self._pred_edge_output = None


    def forward(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        #print(x.shape)
        # print(self.node_emb)
        self._test_edge_embedding = False
        if self.use_seperate_latent_embedding:
            skill_label = (int)(action.cpu().detach().numpy()[0][0])
            # print('skill_label', skill_label)
        if self.use_latent_action:
            # print(action.shape)
            # print(self.action_emb)
            if self.use_seperate_latent_embedding:
                action = self.action_emb(action[:, 1:])
            else:
                action = self.action_emb(action)
        if self.node_embedding:
            x = self.node_emb(x)

        # Begin the message passing scheme
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        
        # print(self._per_node_output_size)
        if self.use_seperate_latent_embedding:
            graph_node_action = torch.cat((state_pred_out, action), axis = 1)
            if skill_label == 0:
                pred_node_embedding = self.graph_dynamics_0(graph_node_action)
            elif skill_label == 1:
                pred_node_embedding = self.graph_dynamics_1(graph_node_action)
            # pred_node_embedding = self.graph_dynamics[skill_label](graph_node_action)

            
            edge_num = self._pred_edge_output.shape[0]
            edge_action_list = []
            for _ in range(edge_num):
                edge_action_list.append(action[0][:])
            edge_action = torch.stack(edge_action_list)
            graph_edge_node_action = torch.cat((self._pred_edge_output, edge_action), axis = 1)
            if skill_label == 0:
                pred_graph_edge_embedding = self.graph_edge_dynamics_0(graph_edge_node_action)
            elif skill_label == 1:
                pred_graph_edge_embedding = self.graph_edge_dynamics_1(graph_edge_node_action)
            # pred_graph_edge_embedding = self.graph_edge_dynamics[skill_label](graph_edge_node_action)
        else:
            graph_node_action = torch.cat((state_pred_out, action), axis = 1)
            pred_node_embedding = self.graph_dynamics(graph_node_action)

            
            edge_num = self._pred_edge_output.shape[0]
            edge_action_list = []
            for _ in range(edge_num):
                edge_action_list.append(action[0][:])
            edge_action = torch.stack(edge_action_list)
            graph_edge_node_action = torch.cat((self._pred_edge_output, edge_action), axis = 1)
            pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        
        return_dict = {'pred': state_pred_out,
        'current_embed': state_pred_out, 'pred_embedding':pred_node_embedding, 'edge_embed': self._pred_edge_output, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            #print(self._pred_edge_output_sigmoid)
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        # if self._use_edge_dynamics:
        #     return_dict['dynamics_edge'] = dynamics_edge

        return return_dict

    def forward_decoder(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        # print(x.shape)
        # print(self.node_emb)
        #x = self.node_emb(x)
        

        # Begin the message passing scheme
        self._test_edge_embedding = True
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        #print(state_pred_out.shape)
        # print(state_pred_out.shape)
        # print(action.shape)
        state_action = torch.cat((state_pred_out, action), axis = 1)
        #print(state_action.shape)
        pred_state = self.dynamics(state_action)

        # print(self._pred_edge_output.shape)
        # print(action.shape)
        edge_action = torch.zeros((self._pred_edge_output.shape[0], self._pred_edge_output.shape[1] + self.action_dim))
        edge_action[:,:self._pred_edge_output.shape[1]] = self._pred_edge_output
        edge_action[:,self._pred_edge_output.shape[1]:] = action[0]
        edge_action = edge_action.to(x.device)
        #print(edge_action)

        #edge_action = torch.cat((self._pred_edge_output, action), axis = 1)
        #print(state_action.shape)
        
        if self._use_edge_dynamics:
            dynamics_edge = self.edge_dynamics(edge_action)

        graph_node_action = torch.cat((x, action), axis = 1)
        pred_node_embedding = self.graph_dynamics(graph_node_action)

        #edge_action = torch.stack([action[0][:], action[0][:], action[0][:], action[0][:], action[0][:], action[0][:]])
        edge_num = self._edge_inp.shape[0]
        edge_action_list = []
        for _ in range(edge_num):
            edge_action_list.append(action[0][:])
        edge_action = torch.stack(edge_action_list)
        graph_edge_node_action = torch.cat((self._edge_inp, edge_action), axis = 1)
        pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        return_dict = {'pred': state_pred_out, 'object_mask': mask_out, 'graph_pred': graph_preds, 'pred_state': pred_state, 
        'current_embed': x, 'pred_embedding':pred_node_embedding, 'edge_embed': self._edge_inp, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        if self._use_edge_dynamics:
            return_dict['dynamics_edge'] = dynamics_edge

        return return_dict

    
    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr is the edge attribute between x_i and x_j

        # x_i is the central node that aggregates information
        # x_j is the neighboring node that passes on information.

        # Concatenate features for sender node (x_j) and receiver x_i and get the message from them
        # Maybe there is a better way to get this message information?

        if self._test_edge_embedding:
            edge_inp = edge_attr
        else:
            if self._use_edge_embedding:
                assert self.edge_emb is not None, "Edge embedding model cannot be none"
                # print(edge_attr.shape)
                # print(self.edge_emb)
                edge_inp = self.edge_emb(edge_attr)
            else:
                edge_inp = edge_attr
        self._edge_inp = edge_inp
        #print('edge in GNN', self._edge_inp)

        #print(edge_inp.shape)
        if self.use_edge_input:
            x_ij = torch.cat([x_i, x_j, edge_inp], dim=1)
            # print(x_ij.shape)
            # print(self.message_info_mlp)
            out = self.message_info_mlp(x_ij)
        else:
            x_ij = torch.cat([x_i, x_j], dim=1)
            # print(x_ij.shape)
            # print(self.message_info_mlp)
            out = self.message_info_mlp(x_ij)
        #print('out', out.shape)
        # print(out)
        return out

    def update(self, x_ij_aggr, x, edge_index, edge_attr):
        # We can transform the node embedding, or use the transformed embedding directly as well.
        inp = torch.cat([x, x_ij_aggr], dim=1)
        if self._should_predict_edge_output:
            source_node_idxs, target_node_idxs = edge_index[0, :], edge_index[1, :]
            if self.use_edge_input:
                edge_inp = torch.cat([
                    self._edge_inp,
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            else:
                edge_inp = torch.cat([
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            # print(edge_inp.shape)
            # print(self._edge_output_sigmoid)
            # print(self._edge_output_mlp)
            self._pred_edge_output = self._edge_output_mlp(edge_inp)
            self._pred_edge_output_sigmoid = self._edge_output_sigmoid(edge_inp)
            #print(self._pred_edge_output_sigmoid)
            if self.all_classifier:
                self._pred_edge_classifier = []
                for pred_classifier in self.all_classifier_list:
                    pred_classifier = pred_classifier.to(x.device)
                    self._pred_edge_classifier.append(F.softmax(pred_classifier(edge_inp), dim = 1))
        # print('x, x_ij_aggr', [x.shape, x_ij_aggr.shape])
        # print(x_ij_aggr)
        return self.node_output_mlp(inp)

    def edge_decoder_result(self):
        if self._should_predict_edge_output:
            return self._pred_edge_output
        else:
            return None

