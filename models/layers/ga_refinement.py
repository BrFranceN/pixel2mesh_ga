

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
# from models.layers.ga_transformer import GA_transformer  # Only GATR version
# from gatr.interface import embed_point, extract_scalar
from models.layers.mlp_deformer import VertexDeformer

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
        self.unpooling = deepcopy((source_model.unpooling))

        
        

        self.gcns = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                ellipsoid.adj_mat[2], activation=self.gconv_activation,use_mod=None )
        
        self.gcns = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                ellipsoid.adj_mat[2], activation=self.gconv_activation,use_mod=None )
        
        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])



        if gcn2_sd is not None:
            print("preload gcn2_sd ")
            self.gcns.load_state_dict(gcn2_sd)

        if gconv_sd is not None:
            print("preload gconv_sd ")
            self.gconv.load_state_dict(gconv_sd)




    def forward(self,out_pretrained,img):
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        x2 = out_pretrained['pred_coord'][1]
        x_hidden = out_pretrained['my_var'][0]



        # GCN Block 3
        x_unpooling_3 = self.projection(img_shape, img_feats, x2)
        # x_tmp = x #questo va riottenuto!
        x_unpooling_3 = self.unpooling[1](torch.cat([x_unpooling_3, x_hidden], 2))
        x3, _ = self.gcns(x_unpooling_3)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)




        return x3








class refinement(nn.Module):
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
        super(refinement,self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim
        self.gconv_activation = gconv_activation
        self.features_dim = features_dim

        self.gconv_initial = GConv(in_features=self.features_dim + self.hidden_dim + 8, out_features=self.features_dim + self.hidden_dim,adj_mat=ellipsoid.adj_mat[1])
        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])
        

        self.gcns = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                ellipsoid.adj_mat[2], activation=self.gconv_activation)

        if gcn2_sd is not None:
            print("preload gcn2_sd ")
            self.gcns.load_state_dict(gcn2_sd)

        if gconv_sd is not None:
            print("preload gconv_sd ")
            self.gconv.load_state_dict(gconv_sd)


    def forward(self,x3):
        new_x3, new_hidden =  self.gcns(x3)

        new_x3 = F.relu(new_x3)
        new_x3 = self.gconv(new_x3)

        return new_x3
        
        