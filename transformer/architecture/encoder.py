# ----------------------- encoding utf-8 --------------------------------
# defining encoder block for transformer architecture
# all the layer from utils.py


import torch
import os
import sys
import torch.nn as nn
from utils import EmbeddingLayer , LayerNormalization , PositionalEncodingLayer , FeedForwardLayer , ResidualConnectionLayer
from attention import MultiHeadAttention


"""
    defining encoder block for transformer architecture.
    as given in paper attention all you need : https://arxiv.org/pdf/1706.03762
"""
class EncoderBlock(nn.Module) :

    def __init__(self , dim :int , num_head :int ,input_embedding_shape :int , f_dim :int , s_dim :int ,  dropout :float , bias :bool) -> None:
        super(EncoderBlock , self).__init__()
        self.dim = dim
        self.num_heads = num_head
        self.dropout = dropout
        self.bias_opt = bias
        self.input_embedding_shape = input_embedding_shape
        self.f_dim = f_dim
        self.s_dim = s_dim
        self.layernorm = LayerNormalization(dim = self.dim)
        self.feed_forward = FeedForwardLayer(self.input_embedding_shape , self.f_dim  , self.s_dim)
        self.residual_block = ResidualConnectionLayer()
        self.self_attention = MultiHeadAttention(self.dim , self.num_heads , self.dropout , self.bias_opt)



    def forward(self , input_embedding :torch.Tensor , input_mask :torch.Tensor):
        # shape of input_embedding should be batch , seq_length , dim
        out , _ = self.self_attention(input_embedding , input_embedding , input_embedding , input_mask)
        out = self.residual_block(out , self.feed_forward)

        out = self.layernorm(out)
        out = self.residual_block(out , self.feed_forward)
        out = self.layernorm(out)
        return out

"""
    this class stacks n number of EncoderBlock class.
"""
class EncoderLayer(nn.Module):

    def __init__(self,num_head , num_layer,bias :bool , dim:int , input_embedding_shape :int , f_dim :int , s_dim :int , dropout : float = 0.1 ) -> None:
        super(EncoderLayer , self).__init__()
        self.dropout = dropout
        self.bias = bias
        self.dim = dim
        self.num_heads = num_head
        self.num_layer = num_layer
        self.drop_layer = nn.Dropout(self.dropout)
        self.f_dim = f_dim
        self.s_dim = s_dim
        self.input_embedding_shape = input_embedding_shape

        self.module_list = nn.ModuleList([EncoderBlock(self.dim , self.num_heads , self.input_embedding_shape , self.f_dim , self.s_dim , self.dropout ,self.bias) for _ in range(self.num_layer)])

    def forward(self , x:torch.Tensor , input_mask : torch.Tensor ):
        out = x
        for modulo in self.module_list:
            out = modulo(out , input_mask)
        return self.drop_layer(out)


# testing encoder layer
if __name__ == "__main__":
    encoder_layer = EncoderLayer(4 , 1 , True , 768 , 768 , 512 , 768 , 0.2)
    inp_tensor = torch.randn(size=(16 , 120 , 768 ))
    out = encoder_layer(inp_tensor , None)
    print("output shape is " ,out.shape )
