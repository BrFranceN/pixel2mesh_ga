import torch.nn as nn
import torch.nn.functional as F

from models.layers.gconv import GConv
# from models.layers.gconv import GAConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None,stampa=False):
        super(GResBlock, self).__init__()

        self.stampa = stampa

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat,stampa=self.stampa)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat,stampa=self.stampa)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):


    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None,use_mod=None,stampa=False):
        super(GBottleneck, self).__init__()

        self.use_mod = use_mod  #TODO REMOVE
        self.stampa = stampa

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation,stampa=self.stampa)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat,stampa=self.stampa)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat,stampa=self.stampa)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        # print(f"x in gconv -> {x.shape}")
        if self.activation:
            x = self.activation(x)
        if self.use_mod is not None:
            for i in range(self.use_mod):
                x_hidden = self.blocks[i](x)
        else:
            x_hidden = self.blocks(x)
        # print(f"x hidden  in gconv -> {x_hidden.shape}")
        x_out = self.conv2(x_hidden)
        # print(f"x_out  in gconv -> {x_out.shape}")

        return x_out, x_hidden
    



# class GA_GResBlock(nn.module):
#     def __init__(self,in_dim,hidden_dim,adj_mat,activation=None):
#         super(GA_GResBlock,self).__init__()

#         self.conv1 = GA_GConv(in_features=in_dim, out_features=hidden_dim,adj_mat=adj_mat)
#         self.conv2 = GA_GConv(in_features=hidden_dim, out_features=in_dim,adj_mat=adj_mat)
#         self.activation = F.gelu if activation else None

#         def forward(self,inputs):
#             pass





'''
BLOCK 3 ANALYSIS
inputs in gconv:  torch.Size([8, 2466, 4035])
weight in gconv:  torch.Size([4035, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

inputs in gconv:  torch.Size([8, 2466, 192])
weight in gconv:  torch.Size([192, 192])
support in gconv:  torch.Size([8, 2466, 192])
support_loop in gconv:  torch.Size([8, 2466, 192])
output in gconv:  torch.Size([8, 2466, 192])
bias shape  torch.Size([192])

'''
















''' old things to delete 
# import torch
# import torch.nn as nn
# from algebra.cliffordalgebra import CliffordAlgebra

# class CliffordAttention(nn.Module):
#     def __init__(self, in_features, out_features, algebra):
#         super(CliffordAttention, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.algebra = algebra
#         self.linear = nn.Linear(in_features, out_features)
#         self.att_weights = nn.Parameter(torch.Tensor(out_features, out_features))

#     def forward(self, x):
#         # Applica una trasformazione lineare
#         h = self.linear(x)
#         # Calcolo del prodotto Clifford per attenzione
#         attention_scores = torch.einsum('ijk,kl->ijl', h, self.att_weights)
#         attention_scores = self.algebra.geometric_product(attention_scores, attention_scores)
#         # Applicazione dei pesi di attenzione
#         attention = torch.softmax(attention_scores, dim=-1)
#         output = torch.matmul(attention, h)
#         return output
# # class that use geometric algebra
# class

'''