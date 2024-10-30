import torch.nn as nn
import torch.nn.functional as F

from models.layers.gconv import GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim, adj_mat=adj_mat)
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

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden
    

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
