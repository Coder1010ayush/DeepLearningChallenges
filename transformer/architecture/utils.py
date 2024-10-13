# ------------------------- utf-8 encoding --------------------------
import torch
from torch.distributions import Gamma
import torch.nn as nn
import os
import sys
import pathlib
from torch.nn.init import kaiming_normal
from torch.nn.modules.sparse import init
import math

"""
    class input : vocab_size , dim
    vocab_size : number of unique ids in dataset
    dim : vector length

    forward func input : x [ tensor of int values or ids ]
    output : [ len(x) , dim ] matric
            ( dim size of vector for each ids )
"""
class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size :int , dim :int ) -> None:
        super(EmbeddingLayer , self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size > 0 , "it can not be zero"
        assert dim > 0 , "it can not be zero"
        self.embedding_layer = nn.Embedding(self.vocab_size ,self.dim)


    def forward(self , x):
        return self.embedding_layer(x)


"""
    class input : eps value
    eps_value : default 10** -6

    forward func input : x
    output : each row will be normalized according to their mean and std of that row.

"""
class LayerNormalization(nn.Module):

    def __init__(self ,dim :int,  eps :float = 10** -6) -> None:
        super(LayerNormalization , self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim)) # weight term
        self.beta = nn.Parameter(torch.zeros(dim)) # bias term

    def forward(self , x :torch.Tensor):
        init.uniform_(self.gamma ,0 , 1)
        mean = x.mean(dim = -1 , keepdim= True)
        std = x.std(dim = -1 ,keepdim=True) ** 2
        # print('shape of mean is ', mean.shape)
        # print("std shape is ", std.shape)
        x = (x-mean) / (std + self.eps).sqrt()
        out = self.gamma * x + self.beta
        return out

"""
    class input : nothing
    forward input : tensor and sublayer
    output : call the sublayer for input tensor and return it.
"""
class ResidualConnectionLayer(nn.Module):

    def __init__(self , dropout : float = 0.1) -> None:
        super(ResidualConnectionLayer , self ).__init__()
        self.drop_layer = nn.Dropout(p=dropout)

    def forward(self , x , sublayer ):

        return x + self.drop_layer(sublayer(x) )

class FeedForwardLayer(nn.Module):

    def __init__(self,in_feature_dim, f_layer_dim :int , s_layer_dim :int , dropout:float = 0.1) -> None:
        super(FeedForwardLayer , self ).__init__()
        self.f_layer_dim = f_layer_dim
        self.s_layer_dim = s_layer_dim
        self.drop_layer = nn.Dropout(p=dropout)

        self.fc_layer_1 = nn.Linear(in_features=in_feature_dim ,out_features=self.f_layer_dim)
        self.fc_layer_2 = nn.Linear(in_features=self.f_layer_dim , out_features=self.s_layer_dim)
        self.relu = nn.ReLU()

    def forward(self , x:torch.Tensor):
        out = self.fc_layer_1(x)
        out = self.relu(out)
        out = self.fc_layer_2(out)
        out = self.relu(self.drop_layer(x))
        return out

class PositionalEncodingLayer(nn.Module) :
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# testing all these class
if __name__ == "__main__":
    # testing layernormalization class
    inp = torch.randn(size= (16 , 32, 768))
    lnorm = LayerNormalization(dim = 768)
    out = lnorm(inp)
    print("output shape is " , out.shape)
    pos_layer = PositionalEncodingLayer(d_model=768 )
    out = pos_layer( out )
    print("output shape after position encoding layer is ", out.shape)
    print()
    inp = torch.randint(low=10 , high=20 , size=(10,))
    embed_layer = EmbeddingLayer(vocab_size=400 , dim=768)
    out = embed_layer(inp)
    print("output shape is ", out.shape)
