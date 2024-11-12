import torch
import torch.nn as nn
import numpy as np
import sys


class GUnpooling(nn.Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.
    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """

    def __init__(self, unpool_idx):
        super(GUnpooling, self).__init__()
        self.unpool_idx = unpool_idx
        # print(f"unpooo_idx shape -> {unpool_idx.shape}")
        # print(f"unpooo_idx shape -> {unpool_idx}")
        # save dim info
        self.in_num = torch.max(unpool_idx).item()
        self.out_num = self.in_num + len(unpool_idx)

    def forward(self, inputs):
        new_features = inputs[:, self.unpool_idx].clone()
        # print(f"newfeatures ->{new_features.shape}")
        new_vertices = 0.5 * new_features.sum(2) # interpolazione tra le features dei bordi
        # print(f"new_vertices ->{new_vertices.shape}")
        output = torch.cat([inputs, new_vertices], 1)
        # print(f"output ->{output.shape}")
        # sys.exit()
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_num) + ' -> ' \
               + str(self.out_num) + ')'