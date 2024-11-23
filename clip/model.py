# ------------------------------------------ utf-8 encoding --------------------------------------------
# clip model link : https://arxiv.org/pdf/2103.00020

import matplotlib.patches as mpatches
import os
import sys
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append("/home/infinity/Documents/icpr_challenges")


def visualize_clip_model():
    fig, ax = plt.subplots(figsize=(8, 6))
    text_encoder = mpatches.FancyBboxPatch((0.1, 0.6), 0.3, 0.2, boxstyle="round,pad=0.1", ec="black", fc="lightblue")
    ax.add_patch(text_encoder)
    ax.text(0.25, 0.7, 'Text Encoder', ha='center', va='center', fontsize=12)

    image_encoder = mpatches.FancyBboxPatch((0.6, 0.6), 0.3, 0.2, boxstyle="round,pad=0.1", ec="black", fc="lightgreen")
    ax.add_patch(image_encoder)
    ax.text(0.75, 0.7, 'Image Encoder', ha='center', va='center', fontsize=12)

    shared_embedding = mpatches.FancyBboxPatch(
        (0.35, 0.3), 0.3, 0.2, boxstyle="round,pad=0.1", ec="black", fc="lightcoral")
    ax.add_patch(shared_embedding)
    ax.text(0.5, 0.4, 'Shared Embedding Space', ha='center', va='center', fontsize=12)

    ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()

# model arguments


class ModelArgs:
    num_heads = 4
    num_layer = 2
    dim = 768
    img_size = 256
    channel = 3
    patch_size = 16


class Clip(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super(Clip, self).__init__()
        self.args = args
        from trf.architecture.encoder import EncoderLayer
        self.text_encoder = EncoderLayer(num_head=self.args.num_heads,
                                         num_layer=self.args.num_layer, bias=True, dim=self.args.dim, input_embedding_shape=self.args.dim, f_dim=self.args.dim, s_dim=self.args.dim)
        from vision_transformer.vit import Vit
        self.image_encoder = Vit(args=self.args)
        self.similarity_scaling = nn.Parameter(torch.randn(1))

    def forward(self, x_text, x_img):
        text_emd = self.text_encoder(x_text, None)
        img_emd = self.image_encoder(x_img, None)
        text_out = text_emd.mean(dim=1)
        img_out = img_emd.mean(dim=1)
        text_out_norm = text_out / text_out.norm(dim=1, keepdim=True)
        img_out_norm = img_out / img_out.norm(dim=1, keepdim=True)

        similarity = torch.matmul(text_out_norm, img_out_norm.T) * self.similarity_scaling
        return similarity


if __name__ == "__main__":
    args = ModelArgs()
    text_t = torch.randn(size=(16, 121, 768))
    img_t = torch.randn(size=(16, 3, 256, 256))
    clp = Clip(args=args)
    out = clp(text_t, img_t)
    print("out shape is ", out.shape)
