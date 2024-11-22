

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from models.layers.gpooling import GUnpooling
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.selfattention_ga import SelfAttentionGA
from models.layers.selfattention_ga import TransformerEncoderGA
from algebra.cliffordalgebra import CliffordAlgebra

class ga_refinement(nn.Module):
    def __init__(self,  
        hidden_dim,
        features_dim, 
        coord_dim, 
        last_hidden_dim,
        ellipsoid,
        gconv_activation,
        source_model,
        gcn2_sd = None,
        gconv_sd = None):
        super(ga_refinement,self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim
        self.gconv_activation = gconv_activation
        self.features_dim = features_dim


        self.nn_encoder = deepcopy(source_model.nn_encoder)
        self.nn_decoder = deepcopy(source_model.nn_decoder) # to use in case of debug to visualize features
        self.projection = deepcopy(source_model.projection)
        self.unpooling = deepcopy((source_model.unpooling[1]))


        


        self.gconv_initial = GConv(in_features=self.features_dim + self.hidden_dim + 8, out_features=self.features_dim + self.hidden_dim,adj_mat=ellipsoid.adj_mat[2])
        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])
        

        self.gcns = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                ellipsoid.adj_mat[2], activation=self.gconv_activation)
        





        # print("gcns prima del load ")
        # for name, param in self.gcns.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Shape: {param.size()}")
        #     print(f"Values:\n{param.data}")



        if gcn2_sd is not None:
            print("preload gcn2_sd ")
            self.gcns.load_state_dict(gcn2_sd)

        if gconv_sd is not None:
            print("preload gconv_sd ")
            self.gconv.load_state_dict(gconv_sd)


        # print("gcns dopo il load ")
        # for name, param in self.gcns.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Shape: {param.size()}")
        #     print(f"Values:\n{param.data}")
        
        # print("quello da caricare")
        # for name, param in self.gconv.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Shape: {param.size()}")
            # print(f"Values:\n{param.data}")
        
        #new (indeciso tra questo oppure un MLP)
        # sono queste che hanno parametri trainabili

        # self.gcns = GBottleneck(6, self.features_dim + self.hidden_dim + 8, self.hidden_dim, self.last_hidden_dim,
        #                 ellipsoid.adj_mat[2], activation=self.gconv_activation)

        # self.unpooling = GUnpooling(ellipsoid.unpool_idx[1])
    
        algebra_dim = 3
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        embed_dim = 2**algebra_dim
        hidden_dim = 256
        self.self_attention_ga = SelfAttentionGA(self.algebra,embed_dim,hidden_dim)
        self.transformerEncoding= TransformerEncoderGA(self.algebra,embed_dim, hidden_dim,1)


    def forward(self,x2,x_hidden,img):
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)
        x = self.projection(img_shape, img_feats, x2)



        x2_mv = self.algebra.embed_grade(x2,1)
        x2_result = self.transformerEncoding(x2_mv)
        

        x_new = self.unpooling(torch.cat([x, x_hidden,x2_result], 2))

        # print("x_new shape",x_new.shape)
        x_new_2 = self.gconv_initial(x_new)
        # print("x_new2shape",x_new_2.shape)

        x4,x4_hidden_final = self.gcns(x_new_2)

        # print("x4.shape ", x4.shape )

        if self.gconv_activation:
            x4 = F.relu(x4)
        
        x4 = self.gconv(x4)
        # print("x4.shape ", x4.shape )


        return x4

        
        