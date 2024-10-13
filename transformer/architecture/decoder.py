# -------------------------- encoding utf-8 -------------------------------
# defining single decoder block and decoder layer as discussed in the paper
# paper link is : https://arxiv.org/pdf/1706.03762
# all the layers will be used as defined in the utils.py file.


import torch
import torch.nn as nn
import os
import sys
from utils import EmbeddingLayer , LayerNormalization , PositionalEncodingLayer , ResidualConnectionLayer , FeedForwardLayer
from attention import MultiHeadAttention


class DecoderBlock(nn.Module):

    def __init__(self , dim :int , num_heads :int , bias_opt :bool ,input_embedding_shape :int, f_dim :int,
        s_dim :int, dropout :float = 0.1) -> None:
        super(DecoderBlock , self).__init__()
        self.num_head = num_heads
        self.dropout = dropout
        self.bias_opt = bias_opt
        self.input_embedding_shape = input_embedding_shape
        self.f_dim = f_dim
        self.s_dim = s_dim
        self.dim = dim

        self.layernorm = LayerNormalization(dim=self.dim )
        self.residual_block = ResidualConnectionLayer(dropout=self.dropout)
        self.feed_forward = FeedForwardLayer(self.input_embedding_shape , self.f_dim , self.s_dim , self.dropout)
        self.self_attention = MultiHeadAttention(dim = self.dim  , num_heads=self.num_head , dropout=self.dropout , bias= self.bias_opt)
        self.cross_attention = MultiHeadAttention(dim = self.dim , num_heads=self.num_head , dropout=self.dropout , bias= self.bias_opt)


    def forward(self ,input_embedding_tensor:torch.Tensor, output_embedding :torch.Tensor , output_mask :torch.Tensor , input_mask :torch.Tensor):

        attn_out , _ = self.cross_attention(output_embedding , output_embedding , output_embedding , output_mask)
        out = self.residual_block(attn_out , self.feed_forward)
        out = self.layernorm(out)

        c_attn_out , _ = self.self_attention(input_embedding_tensor , out , out , input_mask)
        out = self.residual_block(c_attn_out , self.feed_forward)
        out = self.layernorm(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, num_layer , dim :int , num_heads :int , bias_opt :bool ,input_embedding_shape :int, f_dim :int,
        s_dim :int, dropout :float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.num_layer = num_layer
        self.dim = dim
        self.bias_opt = bias_opt
        self.num_heads = num_heads
        self.input_embedding_shape = input_embedding_shape
        self.f_dim = f_dim
        self.s_dim = s_dim
        self.dropout = dropout
        self.drop_layer = nn.Dropout(self.dropout)

        self.moduel_list = nn.ModuleList([DecoderBlock(self.dim , self.num_heads, self.bias_opt , self.input_embedding_shape , self.f_dim , self.s_dim , self.dropout) for _ in range(self.num_layer)])

    def forward(self , input_embedding_tensor :torch.Tensor , output_embedding_tensor :torch.Tensor , input_mask :torch.Tensor , output_mask :torch.Tensor):
        out = output_embedding_tensor
        for modul in self.moduel_list:
            out = modul(input_embedding_tensor , out , output_mask , input_mask)
        return self.drop_layer( out )

# testing decoder layer

if __name__ == "__main__":
    decoder_obj = DecoderLayer(1 , 20 , 2 , True , 20 ,15 , 20 , 0.4 )
    inp = torch.randn(size = (2 , 70  , 20 ))
    out = decoder_obj(inp , inp , None , None)
    print("output shape is " , out.shape)
