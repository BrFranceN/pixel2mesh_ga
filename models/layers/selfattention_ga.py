

from algebra.cliffordalgebra import CliffordAlgebra

import torch
import torch.nn as nn
import torch.nn.functional as F


import math










#TODO TEST SERIO ATTENTION LAYER


#@title Multivector linear layer

class MVLinear(nn.Module):
    """
    Multivector linear layer: add weights and biases to the elements of a multivector.
    """

    def __init__(self, algebra, in_features, out_features):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        max_val = 600

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.bias = nn.Parameter(torch.empty(1, 1, 1))
        self.b_dims = (0,)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        # element-wise multiplication
        # start = time.time()
        weight = self.weight

        result = torch.matmul(input, weight)
        # print(f" input.shape X  weight.shape  ->  {input.shape}X {weight.shape}  ")
        # result = torch.einsum('ijk,kl->ijl', input, weight) # slower alternative
        # end = time.time()
        # print(f"operation took {end - start:.4f} seconds") # single operation time

        """
        # Legacy code used to replace the ensum/matmul but it is slower (do not parallelize)
        start = time.time()
        for b in range(input.shape[0]): # batch size (b)
            for m in range(weight.shape[1]): # embed_dim  (d)
                for i in range(input.shape[-1]): # embed_dim (d)
                    for n in range(weight.shape[0]): # seq len (s)
                        custom[b,m,i] += input[b,m,i] * weight[n,m]

                        #       b m i           n m
                        # input[0,0,0] * weight[0,0]
                        # input[0,1,0] * weight[1,0]
                        #             ...
                        # input[0,s,0] * weight[s,0]

                        # input[0,0,1] * weight[0,1]
                        # input[0,1,1] * weight[1,1]
                        #             ...
                        # input[0,s,1] * weight[s,1]

                        # input[0,0,d] * weight[0,d]
                        # input[0,1,d] * weight[1,d]
                        #             ...
                        # input[0,s,d] * weight[s,d]

                        # input[1,0,0] * weight[0,0]
        end = time.time()
        print(f"for loop took {end - start:.4f} seconds") # single operation time
        """

        # print(f"result shape -> {result.shape}")
        bias = self.algebra.embed(self.bias, self.b_dims)
        # print(f"bias shape -> {bias.shape}")
        result = result + bias
        return result

@torch.jit.script
def fast_einsum(q_einsum, cayley, k_einsum):
    """
    Implementation of the geometric product between two multivectors made with the einsum notation.
    Compiled with jit script for optimization!

    Args:
        q_einsum (torch.Tensor): left multivector
        cayley: look up tabel for the geometric product, it depends on the algebra used.
        k_einsum (torch.Tensor): right multivector.
    """
    return torch.einsum("...i,ijk,...k->...j", q_einsum, cayley, k_einsum)

# Note: this function is taken from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/models/modules/utils.py
def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor, dim=0):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
        dim: int: starting dim, default: 0.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[dim * (slice(None),) + (None,) * n_unsqueezes]
    

EPS = 1e-6

class NormalizationLayer(nn.Module):
    """
    Normalization layer to scale down the elment of a multivector.
    """

    def __init__(self, algebra, features, init: float = 0):
        super().__init__()
        self.algebra = algebra
        self.in_features = features
        max_seq = 3000 # used to cap the parameters (Note: this is not the best approach)

        # this parameter that learn how much to scale the input data
        # in particular the how much scale the norm of input (see forward)
        self.a = nn.Parameter(torch.zeros(max_seq, algebra.n_subspaces) + init)


    def forward(self, input):
        ("Normalization layer")
        assert input.shape[2] == self.in_features # small change to take in account batch size extra dimention
        # print(f"input.shape => {input.shape}")

        norms = torch.cat(self.algebra.norms(input), dim=-1)
        # print(f"norms.shape  before => {norms.shape}")
        s_a = torch.sigmoid(self.a)
        # print(f"s_a.shape => {s_a.shape}")
        norms = s_a[:input.shape[1], :] * (norms - 1) + 1  # interpolates between 1 and the norm
        # print(f"norms.shape  after => {norms.shape}")

        # when you see repeat_interleave usually means that
        # the same thing is repeated for each subspace
        norms = norms.repeat_interleave(self.algebra.subspaces, dim=-1)
        # print(f"norms.shape  after repeat interleave=> {norms.shape}")
        normalized = input / (norms + EPS)
        return normalized

class FullyConnectedSteerableGeometricProductLayer(nn.Module):
    def __init__(
        self,
        algebra,
        features,
    ):
        """
        Fully connected steerable geometric product layer: a nn Module used to compute pairwise geometric products
        between multivectors of a seme input sequence (all combinations).

        Args:
            agebra: Geometric algebra object
            features: The number of features for the geometric product layer
        """
        super().__init__()
        self.algebra = algebra
        self.features = features

        self.normalization = NormalizationLayer(algebra, features)
        self.q_prj = MVLinear(algebra, features, features)
        self.k_prj = MVLinear(algebra, features, features)

    # @torch.jit.script
    def forward(self, input):
        batch, seq, dim = input.shape

        # mv projection
        q = self.q_prj(input)
        k = self.k_prj(input)

        # mv normalization
        q = self.normalization(q)
        k = self.normalization(k)

        # dimention adjustments
        cayley = self.algebra.cayley.to(input.device) # [dim, dim, dim]
        q_einsum = q.unsqueeze(2)  # [batch, seq, 1, dim]
        k_einsum = k.unsqueeze(1)  # [batch, 1, seq, dim]

        # make tensor contigous in memory for performance optimization
        q_einsum = q_einsum.contiguous()
        k_einsum = k_einsum.contiguous()
        cayley = cayley.contiguous()

        # half precision for performance optimization
        q = q.half()
        k = k.half()
        cayley = cayley.half()

        # serve as context managers or decorators that allow regions
        # of the script to run in mixed precision
        with torch.amp.autocast('cuda'):
            output = fast_einsum(q_einsum, cayley, k_einsum)

        """
        # comment the previous 2 line and uncomment this to monitor time on gpu
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True
        ) as prof:
            with torch.amp.autocast('cuda'):
                output = fast_einsum(q_einsum, cayley, k_einsum)
                output = torch.einsum("...i,ijk,...k->...j", q_einsum, cayley, k_einsum)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        """

        """
        # legacy code to expand the combined gp b.t.w. multivectors of the same sequence with for loops
        output = torch.zeros(batch, seq, seq, dim, device=input.device)
        print(output.shape)
        for b in range(input.shape[0]):
            for m in range(input.shape[1]):
                for n in range(input.shape[1]):
                    output[b,m,n, :] += torch.einsum("...i,ijk,...k->...j", q[b,m,:], cayley, k[b,n,:])
        """

        return output



class GeometricProductAttention(nn.Module):
    def __init__(self, algebra, embed_dim):
        """
        Self-Attention layer using geometric algebra operation.

        Args:
            algebra: Geometric algebra object
            features: The number of features for the geometric product layer
        """
        super(GeometricProductAttention, self).__init__()

        self.algebra = algebra
        self.subspaces_dims = algebra.subspaces
        self.gp_layer = FullyConnectedSteerableGeometricProductLayer(algebra, embed_dim)

        # define projection layers for each subspace (legacy)
        # self.layers = nn.ModuleList([
        #     nn.Linear(self.subspaces_dims[_], 1) for _ in range(self.algebra.dim)
        # ])
        # self.att_prj = nn.Linear(algebra.dim, 1)

        # single projection layer to learn common propertires
        self.att_prj = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # compute pairwise geometric products using the geometric product layer
        # start = time.time()
        new_mv = self.gp_layer(x)

        """
        # legacy code to project multivector into a scalre space to obtain attention scores
        tensor_list = []
        i, j, k = 0, 1, 1
        mv_chunks = copy.copy(new_mv)
        for layer in self.layers:
            # print(f"controls i: {i}, j: {j}, k: {k}, ss: {self.subspaces_dims[k]}")
            # print(f"input shape {mv_chunks[:, :, i:j].shape}")
            # print(f"layer shape {layer}")
            new_mv = layer(mv_chunks[:, :, :, i:j])
            i = j
            j += int(self.subspaces_dims[k].item())
            k += 1
            # print(f"new multivector shape: {new_mv.shape}")
            tensor_list.append(new_mv)
        """

        # result = torch.cat(tensor_list, dim=3)
        # print(f"result shape: {result.shape}")

        # apply attention score projection
        output = self.att_prj(new_mv.float())

        # end = time.time()
        # print(f"attention score computation in {end - start:.4f} seconds") # attention operation time

        return output


class SelfAttentionGA(nn.Module):
    def __init__(self, algebra, embed_dim):
        super(SelfAttentionGA, self).__init__()

        self.algebra = algebra
        self.ga_attention = GeometricProductAttention(algebra, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()
        v = self.v_proj(x)

        # compute attention scores using geometric product
        attn_scores = self.ga_attention(x).squeeze(-1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(batch_size,seq_length,seq_length)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # apply attention to values tensor
        return torch.einsum("bqk,bvd->bqd", attn_probs, v)
