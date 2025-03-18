
import torch
import torch.nn as nn
import math
import time


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # linear projections for queries, keys, and values
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()

        # linear projection and split into q, k, v
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # compute attention scores
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # apply the attention mask by adding a large negative value to masked positions
            # explanation: maybe I'm apply an infinite negative number to not pay attention to that
            # vector that represents the dense embedding of id that are not meaningfull (so attention mask = 0)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size,num_heads,seq_length,seq_length)
            # print(f"attention_mask.shape -> {attention_mask.shape}")
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        """
        # einsum loop expantion (just for analysis do not uncomment!!!)
        for b in range(attn_weights.shape[0]):
            for q in range(attn_weights.shape[2]):
                for h in range(attn_weights.shape[1]):
                    for d in range(v.shape[3]):
                        sum = 0
                        for k in range(v.shape[1]):
                            sum += attn_weights[b,h,q,k] * v[b,k,h,d]
                        print(sum)
        """
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # compute weighted values
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.contiguous().view(batch_size, seq_length, embed_dim)
        return self.out_proj(attn_output)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # multi-head attention and feed-forward layers
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForwardNetwork(embed_dim, ff_dim)

        # layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # dropout layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # self-attention block with residual connection
        attn_output = self.self_attn(x, attention_mask=attention_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # feed-forward block with residual connection (#TODO CAMBIARE CON GRAPH NEURAL NETWOTRK RESIDUAL)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        print("a fine tranformer ",x.shape)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim
            ) for _ in range(num_layers)
        ])
        # self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class MeshDeformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers = 4, num_classes=2, timing=False):
        super(MeshDeformer, self).__init__()

        # encoder + classification head: a simple linear layer for sentiment classification
        self.transformer_encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.set_timing = timing
        self.timing = []

    def forward(self, input_ids, attention_mask=None):
        start = time.time()
        encoder_output = self.transformer_encoder(input_ids, attention_mask=attention_mask)
        cls_token_output = encoder_output[:, 0, :]  # taking the output for the [CLS] token

        end = time.time()
        if self.set_timing:
            self.timing.append(end - start)



        # pass through the classifier head to get class logits
        return self.classifier(cls_token_output)



if __name__=="__main__":
    prova = torch.rand(8,2466,3)
    input_points = 2466
    input_dim = 3
    output_points = 9000

    upsampler = nn.Sequential(
    nn.Linear(input_points * input_dim, output_points * input_dim),  # Map all input points to output points
    nn.ReLU(),
    nn.Linear(output_points * input_dim, output_points * input_dim)  # Fine-tune the upsampling
    )


    batch_size = x.shape[0]
    # Flatten the input for MLP processing
    x = x.view(batch_size, -1)
    x = self.upsampler(x)
    # Reshape to final shape
    x = x.view(batch_size, self.output_points, self.input_dim)



    funziona = upsampler(prova)
    # funziona = upsampler(prova)
    print(funziona)
    # embed = nn.Embedding(2466,128)
    # prova2 = embed(prova)
    # print(prova2.shape)
    # transformer_test =  MeshDeformer()

    # print("ciao")