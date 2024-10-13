# ------------------------------ encoder utf-8 -----------------------------
# defining whole transformer architecture according to paper
# transformer consists of two major block encoder and decoder layer
# both encoder and decoder layers are defined in the encoder.py and decoder.py


import torch
import torch.nn as nn
import os
import sys
from architecture.experiment import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayer
from utils import EmbeddingLayer , PositionalEncodingLayer

class ModuleArgs:
    # change here according to use case or pass custom Module Args class
    num_encoder_layer = 6
    num_decoder_layer = 6
    num_head_encoder = 12
    num_head_decoder = 12
    encoder_dim = 768
    decoder_dim = 768
    f_dim = 512
    s_dim = 768
    input_embedding_dim = 768
    dropout = 0.1
    bias_opt = True
    vocab_size = 5000


class Transformer(nn.Module):

    def __init__(self , args:ModuleArgs):
        super(Transformer , self ).__init__()
        self.embedding_layer = EmbeddingLayer(args.vocab_size , args.input_embedding_dim)
        self.positional_encoding_layer = PositionalEncoding(args.input_embedding_dim)
        self.encoder = EncoderLayer(args.num_head_encoder , args.num_encoder_layer , args.bias_opt , args.encoder_dim , args.input_embedding_dim , args.f_dim , args.s_dim ,args.dropout)
        self.decoder = DecoderLayer(args.num_decoder_layer , args.decoder_dim , args.num_head_decoder , args.bias_opt , args.input_embedding_dim , args.f_dim , args.s_dim , args.dropout)

    def forward(self , x  , y , x_mask , y_mask ):
        input_embedding =self.embedding_layer(x)
        input_embedding = self.positional_encoding_layer(input_embedding)

        output_embedding = self.embedding_layer(y)
        output_embedding = self.positional_encoding_layer(output_embedding)

        input_embedding_encoder_out = self.encoder(input_embedding , x_mask)
        out_embedding_decoder_out = self.decoder(input_embedding_encoder_out , output_embedding , x_mask , y_mask)

        return out_embedding_decoder_out
