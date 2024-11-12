import os 
import sys # for stopping whenever I want!
sys.path.append('clifford-group-equivariant-neural-networks')

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection
from models.layers.selfattention_ga import SelfAttentionGA



from algebra.cliffordalgebra import CliffordAlgebra
    



class P2MModel(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(P2MModel, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        


        # initialize geometric algebra 
        # TODO makes this global
        algebra_dim = 3
        metric = [1 for _ in range(algebra_dim)]
        self.algebra = CliffordAlgebra(metric)
        embed_dim = 2**algebra_dim



        self.self_attention_ga = SelfAttentionGA(self.algebra,embed_dim)

        # check if the function on elissoids work
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])



        #TEST USATO PER AGGIUNGERE LA GEOMETRIC ALGEBRA ALLA GRAPH CONVOLUTION
        # GBottleneck(6, self.features_dim + self.hidden_dim + 8, self.hidden_dim, self.last_hidden_dim,
        #                 ellipsoid.adj_mat[2], activation=self.gconv_activation)

        
    

        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])
        
        self.gconv_final = GConv(in_features=11,out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)


        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)



        #HERE I OBTAIN NEW COORDINATES
        #MV ATTENTION x1
        # x1_mv = self.algebra.embed_grade(x1,1)
        # x1_att = self.self_attention_ga(x1_mv) 
        # x1_att_vector = self.algebra.get_grade(x1_att,1) #IDEA extraxt only multivector of dim 1 enriched 
        # x1 = x1_att_vector + x1 # NEW x1

        # before deformation 2
        x1_up = self.unpooling[0](x1)
        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1)
 
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)
        # print(f"x_hidden shape -> {x_hidden.shape}")
        # print(f"x2 shape -> {x2.shape}")


 

        #MV ATTENTION x2

        '''

        AGGIUNTA DA VALUTARE
        x2_mv = self.algebra.embed_grade(x2,1)
        x2_att = self.self_attention_ga(x2_mv) 
        '''
        # x2_att_vector = self.algebra.get_grade(x2_att,1)
        # x2 = x2_att_vector + x2 # NEW x2      




        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2)
        # prima = torch.cat([x, x_hidden],2)
        # dopo = torch.cat([x, x_hidden,x2_att],2)
        # print("prima ",prima.shape)
        # print("prima ",dopo.shape)
        # x = self.unpooling[1](torch.cat([x, x_hidden,x2_att], 2)) # Da valutare
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        # print("x qui -- ", x.shape)
        # exit()
        x3, x3_hidden_final = self.gcns[2](x)

        # print("x3  -> ",x3.shape)

        if self.gconv_activation:
            x3 = F.relu(x3)
        
        x3 = self.gconv(x3)
        # print("x3  -> ",x3.shape)



        # after deformation 3
        # x3_mv = self.algebra.embed_grade(x3,1) 
        # x3_att = self.self_attention_ga(x3_mv) 
        # x3_att_vector = self.algebra.get_grade(x3_att,1)
        # x3_concat =torch.cat([x3,x3_att],2) 
        # # 
        # ("x3 att ->", x3_att.shape)
        # # print("x3_concat att ->", x3_concat.shape)
        # x4 = self.gconv_final(x3_concat)
        # print("x4 ->", x4.shape)

        # sys.exit()

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None

        #AGGIUNGO UN ALTRO PEZZO FINALE


        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst,
            "final_hf":x3_hidden_final
        }


#OLD
# class P2MModel(nn.Module):

#     def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
#         super(P2MModel, self).__init__()

#         self.hidden_dim = options.hidden_dim
#         self.coord_dim = options.coord_dim
#         self.last_hidden_dim = options.last_hidden_dim
#         self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
#         self.gconv_activation = options.gconv_activation

#         self.nn_encoder, self.nn_decoder = get_backbone(options)
#         self.features_dim = self.nn_encoder.features_dim + self.coord_dim

#         self.gcns = nn.ModuleList([
#             GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
#                         ellipsoid.adj_mat[0], activation=self.gconv_activation),
#             GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
#                         ellipsoid.adj_mat[1], activation=self.gconv_activation),
#             GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
#                         ellipsoid.adj_mat[2], activation=self.gconv_activation)
#         ])

#         self.unpooling = nn.ModuleList([
#             GUnpooling(ellipsoid.unpool_idx[0]),
#             GUnpooling(ellipsoid.unpool_idx[1])
#         ])

#         # if options.align_with_tensorflow:
#         #     self.projection = GProjection
#         # else:
#         #     self.projection = GProjection
#         self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
#                                       tensorflow_compatible=options.align_with_tensorflow)

#         self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
#                            adj_mat=ellipsoid.adj_mat[2])

#     def forward(self, img):
#         batch_size = img.size(0)
#         img_feats = self.nn_encoder(img)
#         img_shape = self.projection.image_feature_shape(img)

    

#         init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

#         # GCN Block 1
#         x = self.projection(img_shape, img_feats, init_pts)

#         x1, x_hidden = self.gcns[0](x)
#         # x1 -> new coordinate of vertices => ex: torch.Size([8, 156, 3])
#         # x_hidden -> feature learned during gcn => ex:  torch.Size([8, 156, 192])


#         # before deformation 2
#         x1_up = self.unpooling[0](x1)
       
#         # GCN Block 2
#         x = self.projection(img_shape, img_feats, x1)
      
#         x = self.unpooling[0](torch.cat([x, x_hidden], 2))
#         # after deformation 2
#         x2, x_hidden = self.gcns[1](x)
   

#         # before deformation 3
#         x2_up = self.unpooling[1](x2)
    
#         # GCN Block 3
#         x = self.projection(img_shape, img_feats, x2)
   
#         x = self.unpooling[1](torch.cat([x, x_hidden], 2))
       
#         x3, _ = self.gcns[2](x)
    
#         if self.gconv_activation:
#             x3 = F.relu(x3)

        
#         # after deformation 3
#         x3 = self.gconv(x3)
   
#         # sys.exit()

#         if self.nn_decoder is not None:
#             reconst = self.nn_decoder(img_feats)
#         else:
#             reconst = None

#         return {
#             "pred_coord": [x1, x2, x3],
#             "pred_coord_before_deform": [init_pts, x1_up, x2_up],
#             "reconst": reconst
#         }