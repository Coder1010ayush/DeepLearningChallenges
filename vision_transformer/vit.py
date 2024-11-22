# ------------------------------ utf-8 encoding --------------------------------------
# vit's research paper link : https://arxiv.org/pdf/2010.11929
import os
import sys
import torch
from torch import nn
from torch.nn import Conv2d, Linear
from torch.nn import LayerNorm
sys.path.append("/home/infinity/Documents/icpr_challenges")


class ModelArgs:
    num_heads = 4
    num_layer = 2
    dim = 768
    img_size = 256
    channel = 3
    patch_size = 16
    classes = 10
    hidden_dim_layer_1 = 512
    hidden_dim_layer_2 = 196


class PatchEncoder(nn.Module):

    def __init__(self, img_size=256, patch_size=16, channels=3, embedd_dim=768) -> None:
        super(PatchEncoder, self).__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.num_patch = (self.image_size // self.patch_size) ** 2
        self.conv = Conv2d(in_channels=channels, out_channels=embedd_dim,
                           kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        return out


class Vit(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super(Vit, self).__init__()
        self.args = args
        from trf.architecture.encoder import EncoderLayer
        self.patch_embedding = PatchEncoder(
            img_size=self.args.img_size, patch_size=self.args.patch_size, channels=self.args.channel, embedd_dim=self.args.dim)
        self.encoder = EncoderLayer(num_head=self.args.num_heads, num_layer=self.args.num_layer, bias=True,
                                    dim=self.args.dim, input_embedding_shape=self.args.dim, f_dim=self.args.dim, s_dim=self.args.dim)
        self.mlp = MLP(args=self.args)

    def forward(self, x, mask=None):

        patch_embedding = self.patch_embedding(x)
        out = self.encoder(patch_embedding, mask)
        out = self.mlp(out)
        return out


class MLP(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super(MLP, self).__init__()
        self.args = args
        self.fc_layer_1 = nn.Linear(self.args.dim, self.args.hidden_dim_layer_1)
        self.act_func = nn.GELU()
        self.fc_layer_2 = nn.Linear(self.args.hidden_dim_layer_1, self.args.hidden_dim_layer_2)
        self.fc_layer_3 = nn.Linear(self.args.hidden_dim_layer_2, self.args.classes)

    def forward(self, x):
        x = x[:, 0, :]
        out = self.fc_layer_1(x)
        out = self.act_func(out)
        out = self.fc_layer_2(out)
        out = self.act_func(out)
        out = self.fc_layer_3(out)
        return out


if __name__ == "__main__":
    img = torch.randn(size=(16, 3, 256, 256))
    args = ModelArgs
    vit = Vit(args=args)
    out = vit(img)
    print("out shape is ", out.shape)
