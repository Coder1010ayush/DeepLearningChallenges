# ----------------------------------- utf-8 encoding --------------------------------------
# self attention and multihead attention both is implemented here according to paper
# attention all you need paper link : https://arxiv.org/pdf/1706.03762

# import torch
# import torch.nn as nn
# import os
# import sys

# class MultiHeadAttention(nn.Module):

#     def __init__(self, dim :int , num_heads :int , dropout :float , bias :bool) -> None:
#         super(MultiHeadAttention , self ).__init__()
#         self.num_head = num_heads
#         self.dim = dim
#         self.dropout_layer = nn.Dropout(p=dropout)
#         self.bias_opt = bias

#         assert self.dim % self.num_head == 0, f"d_model {self.dim} must be divisible by num_heads {self.num_head}."

#         self.query_projection = nn.Linear(self.dim , self.dim , self.bias_opt)
#         self.key_projection = nn.Linear(self.dim , self.dim , self.bias_opt)
#         self.value_projection = nn.Linear(self.dim , self.dim , self.bias_opt)
#         self.out_projection = nn.Linear(self.dim , self.dim  , self.bias_opt )
#         self.softmax = nn.Softmax(dim = - 1)

#     @staticmethod
#     def attention(query , key , value , mask : None , dropout):

#         attn_score = (query @ key.transpose(-2 , -1 )) / (query.shape[-1])
#         if mask :
#             attn_score = attn_score.masked_fill__(mask == 0 , 10 ** -9)

#         attn_out =value.transpose(-2 , -1) @ attn_score.softmax(dim = -1)
#         return dropout(attn_out) , attn_score

#     def forward(self , query :torch.Tensor , key :torch.Tensor , value :torch.Tensor , mask :torch.Tensor):
#         q_proj = self.query_projection(query)
#         k_proj = self.key_projection(key)
#         v_proj = self.value_projection(value)

#         # shape of q_proj and k_proj and v_proj is same and is batch_size , seq_length , dim
#         batch , seq_length , dim = q_proj.shape
#         q_split_val = q_proj.view(batch , seq_length , self.num_head , self.dim // self.num_head).transpose(1 , 2)
#         k_split_val = k_proj.view(batch , seq_length , self.num_head , self.dim // self.num_head).transpose(1 , 2)
#         v_split_val = v_proj.view(batch , seq_length , self.num_head , self.dim // self.num_head).transpose(1 , 2)

#         attn_out , attn_weight = MultiHeadAttention.attention(q_split_val , k_split_val , v_split_val , None , self.dropout_layer)
#         # print("attention weight shape is ", attn_weight.shape)
#         # print("attention out shape is ", attn_out.shape)
#         # print("attn final out is ", attn_final_out.shape)

#         attn_final_out = attn_out.view(batch , self.dim , seq_length ).transpose(-2 , -1)
#         out = self.out_projection(attn_final_out)
#         return out , attn_weight

# if __name__ == "__main__":
#     attn_obj = MultiHeadAttention(dim=768 , num_heads=4 , dropout=0.1 , bias=True)
#     inp = torch.randn(size=(16 , 120 , 768))
#     inp_2 = torch.randn(size = (16 , 125 , 768))
#     out = attn_obj(inp , inp_2 , inp_2 , None)
#     print("output shape is " , out.shape)


import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dropout_layer = nn.Dropout(p=dropout)
        self.bias_opt = bias

        assert self.dim % self.num_heads == 0, f"d_model {self.dim} must be divisible by num_heads {self.num_heads}."

        self.query_projection = nn.Linear(self.dim, self.dim, self.bias_opt)
        self.key_projection = nn.Linear(self.dim, self.dim, self.bias_opt)
        self.value_projection = nn.Linear(self.dim, self.dim, self.bias_opt)
        self.out_projection = nn.Linear(self.dim, self.dim, self.bias_opt)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor , dropout: nn.Module ):
        attn_score = (query @ key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_score = attn_score.softmax(dim=-1)
        attn_out = attn_score @ value
        return dropout(attn_out) if dropout else attn_out, attn_score

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        q_proj = self.query_projection(query)
        k_proj = self.key_projection(key)
        v_proj = self.value_projection(value)

        assert q_proj.shape[-1] == k_proj.shape[-1] == v_proj.shape[-1] == self.dim, "Projection output dimension mismatch"

        batch_size, seq_length_q, _ = q_proj.shape
        _, seq_length_k, _ = k_proj.shape
        _, seq_length_v, _ = v_proj.shape

        head_dim = self.dim // self.num_heads

        q_split_val = q_proj.view(batch_size, seq_length_q, self.num_heads, head_dim).transpose(1, 2)
        k_split_val = k_proj.view(batch_size, seq_length_k, self.num_heads, head_dim).transpose(1, 2)
        v_split_val = v_proj.view(batch_size, seq_length_v, self.num_heads, head_dim).transpose(1, 2)

        attn_out, attn_weight = MultiHeadAttention.attention(q_split_val, k_split_val, v_split_val, mask, self.dropout_layer)

        attn_final_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_length_q, self.dim)
        out = self.out_projection(attn_final_out)
        return out, attn_weight

if __name__ == "__main__":
    attn_obj = MultiHeadAttention(dim=768, num_heads=4, dropout=0.1, bias=True)
    inp = torch.randn(size=(16, 120, 768))  # Shape for query
    inp_2 = torch.randn(size=(16, 125, 768))  # Shape for key and value
    out, attn_weights = attn_obj(inp, inp_2, inp_2, None)
    print("Output shape is", out.shape)
    print("Attention weights shape is", attn_weights.shape)
