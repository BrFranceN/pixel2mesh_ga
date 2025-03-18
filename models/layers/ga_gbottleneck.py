import sys
print("sys.path: ", sys.path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

from utils.mesh import Ellipsoid

from copy import deepcopy

from models.layers.gpooling import GUnpooling
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv


# import gatr
# import os

# print(f"GATR Path: {gatr.__file__}")
# print("Baselines Directory Contents:", os.listdir(os.path.join(os.path.dirname(gatr.__file__), "baselines")))

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
from models.layers.gcan import GCAMLP,GCAGNNLayer,GCAGNN #TODO soluzione temporanea all'import non funzionante





class GA_GBottleneck(nn.Module):
    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GA_GBottleneck, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.edge_index, self.edge_weight = self.sparse_adjacency_to_edge_index(adj_mat)

        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        self.activation = F.relu if activation else None



        #TODO TEST
        vertex_dim = 3


        self.gcagnn = GCAGNN(
            in_channels=vertex_dim,  # Canali di input per vertice
            out_channels=vertex_dim, # Presumiamo che l'output abbia la stessa dimensione dei vertici
            node_channels=128,       # Dimensioni nascoste intermedie
            message_channels=64,     # Canali nei messaggi GNN
            mlp_hidden_channels=64,  # Canali nascosti nell'MLP
            mlp_hidden_layers=3,     # Numero di livelli nascosti nell'MLP
            message_passing_steps=3  # Numero di passaggi nel message passing
        )






    def forward(self, inputs):
        risultato = self.gcagnn(inputs,self.edge_index)
        print("risultato shape", risultato.shape)

        exit()


    
    def sparse_adjacency_to_edge_index(self,sparse_adj_mat):
        edge_index = sparse_adj_mat.coalesce().indices()  # Ottieni gli indici dei bordi
        edge_weight = sparse_adj_mat.coalesce().values()  # Ottieni i pesi dei bordi
        return edge_index, edge_weight




#old
# class GA_GBottleneck(nn.Module):
#     def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
#         super(GA_GBottleneck, self).__init__()

#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


#         self.edge_index, self.edge_weight = self.sparse_adjacency_to_edge_index(adj_mat)

#         self.edge_index = self.edge_index.to(self.device)
#         self.edge_weight = self.edge_weight.to(self.device)
#         resblock_layers = [
#             GCAGNNLayer(
#                 in_channels=hidden_dim,
#                 out_channels=hidden_dim,
#                 message_channels=hidden_dim,
#                 mlp_hidden_channels=hidden_dim,
#                 mlp_hidden_layers=2,  # O un valore scelto
#             )
#             for _ in range(block_num)
#         ]
#         self.blocks = nn.Sequential(*resblock_layers)
#         self.conv1 = GCAGNNLayer(
#             in_channels=in_dim,
#             out_channels=hidden_dim,
#             message_channels=hidden_dim,
#             mlp_hidden_channels=hidden_dim,
#             mlp_hidden_layers=2,
#         )
#         self.conv2 = GCAGNNLayer(
#             in_channels=hidden_dim,
#             out_channels=out_dim,
#             message_channels=hidden_dim,
#             mlp_hidden_channels=hidden_dim,
#             mlp_hidden_layers=2,
#         )
#         self.activation = F.relu if activation else None
#     def forward(self, inputs):
#         # Inputs expected to be in shape [batch_size, num_nodes, num_features]
#         batch_size, num_nodes, num_features = inputs.size()
#         x = inputs.view(-1, num_features)  # Flatten inputs for processing
#         # Calculate batch-specific edge indices
#         batch_edge_index = self.calculate_batch_edge_index(batch_size, num_nodes)

#         # Process through GCAGNN layers
#         for layer in self.layers:
#             x = layer(x, batch_edge_index)
#             if self.activation:
#                 x = self.activation(x)

#         # Reshape x back to the batched format
#         return x.view(batch_size, num_nodes, -1)

#     def calculate_batch_edge_index(self, batch_size, num_nodes):
#         offsets = (torch.arange(batch_size, device=self.device) * num_nodes).unsqueeze(0).repeat(2, 1)
#         return self.edge_index.repeat(1, batch_size) + offsets







#     def sparse_adjacency_to_edge_index(self,sparse_adj_mat):
#         edge_index = sparse_adj_mat.coalesce().indices()  # Ottieni gli indici dei bordi
#         edge_weight = sparse_adj_mat.coalesce().values()  # Ottieni i pesi dei bordi
#         return edge_index, edge_weight



