import sys 
sys.path.append('clifford-group-equivariant-neural-networks')

from algebra.cliffordalgebra import CliffordAlgebra

import torch
import torch.nn as nn
import torch.nn.functional as F


import math



class MVLinear(nn.Module):

    def __init__(
        self, 
        algebra, 
        in_features, 
        out_features, 
        subspaces=True, 
        bias=True
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features
        self.subspaces = subspaces
        # self.subspace_dims = algebra.subspaces.tolist()  

        if subspaces:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, algebra.n_subspaces)
                )
            self._forward = self._forward_subspaces
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_features, 1))
            self.b_dims = (0,)
        else:
            self.register_parameter('bias', None)
            self.b_dims = ()

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _forward(self, input):
        return torch.einsum("bm...i, nm->bn...i", input, self.weight)

    def _forward_subspaces(self, input):
        weight = self.weight.repeat_interleave(self.algebra.subspaces, dim=-1)
        # print("--------")
        # print(f"4 input: {input.shape}")
        # print(f"self weight: {self.weight.shape}")
        # print(f"weight: {weight.shape}")
        output = torch.einsum("bm...i, nmi->bn...i", input, weight)
        # print(f"output: {output.shape}")
        return output


    def forward(self, input):
        result = self._forward(input)

        if self.bias is not None:
            bias = self.algebra.embed(self.bias, self.b_dims)
            result += unsqueeze_like(bias, result, dim=2)
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
        self.features = features # prima c'era features


        self.normalization = NormalizationLayer(algebra, features)
        #TODO MODIFY 618 -> NUMBER OF VECTOR
        self.q_prj = MVLinear(algebra, 618, 618)
        self.k_prj = MVLinear(algebra, 618, 618)

    # @torch.jit.script
    def forward(self, input):
        batch, seq, dim = input.shape

        # mv projection
        q = self.q_prj(input)
        k = self.k_prj(input)

        # print("input shape:", input.shape)
        # print("q shape : ",q.shape)
        # print("k shape : ",k.shape)

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
        # print(f"output -> {output.shape}")
        return output



class GeometricProductAttention(nn.Module):
    def __init__(self, algebra, embed_dim,hidden_dim):
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
        # self.att_prj = nn.Linear(embed_dim, 1)
        ##feed forward
        self.att_prj = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # compute pairwise geometric products using the geometric product layer
        # start = time.time()
        # print('new_mv before gp_layer', x.shape)
        new_mv = self.gp_layer(x)
        # print('new_mv after gp_layer', new_mv.shape)
        # return new_mv

        """
        # legacy code to project multivector into a scalar space to obtain attention scores
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
        # print("output att_prj :",output.shape)
        

        # end = time.time()
        # print(f"attention score computation in {end - start:.4f} seconds") # attention operation time

        return output


class SelfAttentionGA(nn.Module):
    def __init__(self, algebra, embed_dim,hidden_dim):
        super(SelfAttentionGA, self).__init__()

        self.algebra = algebra
        self.ga_attention = GeometricProductAttention(algebra, embed_dim,hidden_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)# Before
        # self.v_proj = MVLinear(algebra,618, embed_dim) 

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()
        v = self.v_proj(x) # before [8,618,8] (without projection: [8,8,8])

        # print("v.shape => ",v.shape)

        # print(" v shape ", v.shape)
        # compute attention scores using geometric product
        attn_scores = self.ga_attention(x).squeeze(-1)

        # print("attn_scores -> ",attn_scores.shape)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(batch_size,seq_length,seq_length)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

    
        

        # apply attention to values tensor
        return torch.einsum("bqk,bvd->bqd", attn_probs, v)



class TransformerEncoderLayerGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim):
        super(TransformerEncoderLayerGA, self).__init__()

        self.self_attn = SelfAttentionGA(algebra, embed_dim,hidden_dim)
        # feed forward network
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc_in = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attention_mask=None):
        attn_out = self.self_attn(self.norm1(x), attention_mask)
        x = x + attn_out

        # feed-forward
        ff_out = self.fc_in(x)
        ff_out = self.activation(ff_out)
        ff_out = self.fc_out(ff_out)

        # residual and normalization
        x = x + ff_out
        x = self.norm2(x)
        # we are here yheeee!!!
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # create a long enough position tensor
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # return x + self.pe[:, :x.size(1)] #TODO da rivedere 
        return x 


class TransformerEncoderGA(nn.Module):
    def __init__(self, algebra, embed_dim, hidden_dim, num_layers):
        super(TransformerEncoderGA, self).__init__()

        self.algebra = algebra
        self.layers = nn.ModuleList([
            TransformerEncoderLayerGA(
                algebra, embed_dim,hidden_dim
            ) for _ in range(num_layers)
        ])
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, x, attention_mask=None):
        # x = self.embedding(x)

        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, attention_mask)
        return x