import torch
import torch.nn as nn
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv




class VertexDeformer(nn.Module):

    def __init__(self, in_features=3, out_features =3, hidden_dim=256, mode="batch"):
        super(VertexDeformer, self).__init__()
        self.name = "linear-deformer"
        self.mode = mode
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, vertices):
        return vertices + self.fc(vertices) if self.mode == "batch" else self.fc(vertices) * 0.1


    def get_size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    


class VertexDeformer_v2(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    