

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.gpooling import GUnpooling
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.selfattention_ga import SelfAttentionGA
from algebra.cliffordalgebra import CliffordAlgebra

class ga_refinement(nn.Module):
    def __init__(self,  
        hidden_dim,
        features_dim, 
        coord_dim, 
        last_hidden_dim,
        ellipsoid,
        gconv_activation):
        super(ga_refinement,self).__init__()

        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim
        self.gconv_activation = gconv_activation
        self.features_dim = features_dim


        self.gcns = GBottleneck(6, self.features_dim + self.hidden_dim + 8, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])
        self.unpooling = GUnpooling(ellipsoid.unpool_idx[1])
    
        algebra_dim = 3
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        embed_dim = 2**algebra_dim
        self.self_attention_ga = SelfAttentionGA(self.algebra,embed_dim)


    def forward(self,x,x2,x_hidden):
        x2_mv = self.algebra.embed_grade(x2,1)
        x2_att = self.self_attention_ga(x2_mv) 
        # print(f"x2_att => {x2_att.shape}")
        

        x_new = self.unpooling(torch.cat([x, x_hidden,x2_att], 2))
        x4,x4_hidden_final = self.gcns(x_new)

        if self.gconv_activation:
            x4 = F.relu(x4)
        
        x4 = self.gconv(x4)

        return x4

        
        