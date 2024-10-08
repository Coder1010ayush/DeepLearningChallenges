# ------------------------------ utf-8 encoding -------------------------------
# defining model architecture for as defined in the paper of vggnet
# link of research paper is https://arxiv.org/pdf/1409.1556
# doing some changes in the architecture for lowering computation
import torch
import torchvision
import torch.nn as nn
import os
import sys
from dataclasses import dataclass
from torch.nn import Conv2d


@dataclass
class ModelArgs:
    image_size: int = 224
    channels: int = 3  # rgb
    fc_layer_1_hidden_size: int = 1024
    fc_layer_2_hidden_size: int = 512
    fc_layer_3_hidden_size: int = 100

    kernal_size_1: int = 3
    kernal_size_2: int = 3
    kernal_size_3: int = 3

    stride_size_1: int = 1
    stride_size_2: int = 1
    stride_size_3: int = 1

    stride_size_pool1: int = 2
    stride_size_pool2: int = 2
    stride_size_pool3: int = 2

    padding_size_1: int = 1
    padding_size_2: int = 1
    padding_size_3: int = 1

    filter_size_1: int = 128
    filter_size_2: int = 256
    filter_size_3: int = 512


class DenseNet(nn.Module):

    def __init__(self, input_size, args: ModelArgs) -> None:
        super(DenseNet, self).__init__()
        self.fc_layer_1 = args.fc_layer_1_hidden_size
        self.fc_layer_2 = args.fc_layer_2_hidden_size
        self.fc_layer_3 = args.fc_layer_3_hidden_size
        self.input_size = input_size

        self.fc_layer_block_1 = nn.Linear(in_features=self.input_size, out_features=self.fc_layer_1)
        self.fc_layer_block_2 = nn.Linear(in_features=self.fc_layer_1, out_features=self.fc_layer_2)
        self.fc_layer_block_3 = nn.Linear(in_features=self.fc_layer_2, out_features=self.fc_layer_3)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation_fn = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.activation_fn(self.fc_layer_block_1(x))
        # print("INFO : output of first linear block is ", out.shape)
        out = self.activation_fn(self.fc_layer_block_2(out))
        # print("INFO : output of second linear block is ", out.shape)
        out = self.drop(out)
        out = self.activation_fn(self.fc_layer_block_3(out))
        # print("INFO : output of third linear block is ", out.shape)
        out = self.softmax(out)  # getting probabilities for each class label
        return out


class VGGNet(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super(VGGNet, self).__init__()
        self.filter_size_conv1 = args.filter_size_1
        self.filter_size_conv2 = args.filter_size_2
        self.filter_size_conv3 = args.filter_size_3
        self.input_shape = args.image_size
        self.channel = args.channels

        self.kernal_size_conv1 = args.kernal_size_1
        self.kernal_size_conv2 = args.kernal_size_2
        self.kernal_size_conv3 = args.kernal_size_3

        self.stride_size_pool1 = args.stride_size_pool1
        self.stride_size_pool2 = args.stride_size_pool2
        self.stride_size_pool3 = args.stride_size_pool3

        self.stride_size_conv1 = args.stride_size_1
        self.stride_size_conv2 = args.stride_size_2
        self.stride_size_conv3 = args.stride_size_3

        self.padding_size_conv1 = args.padding_size_1
        self.padding_size_conv2 = args.padding_size_2
        self.padding_size_conv3 = args.padding_size_3

        # defining conv block
        self.conv_block1 = Conv2d(
            in_channels=self.channel, out_channels=self.channel, kernel_size=self.kernal_size_conv1,
            stride=self.stride_size_conv1, padding=self.padding_size_conv1
        )
        self.conv_block2 = Conv2d(
            in_channels=self.channel, out_channels=self.channel, kernel_size=self.kernal_size_conv2,
            stride=self.stride_size_conv2, padding=self.padding_size_conv2
        )
        self.conv_block3 = Conv2d(
            in_channels=self.channel, out_channels=self.channel, kernel_size=self.kernal_size_conv3,
            stride=self.stride_size_conv3, padding=self.padding_size_conv3
        )

        self.maxPool_block1 = nn.MaxPool2d(kernel_size=self.kernal_size_conv1, stride=self.stride_size_pool1)
        self.maxPool_block2 = nn.MaxPool2d(kernel_size=self.kernal_size_conv2, stride=self.stride_size_pool2)
        self.maxPool_block3 = nn.MaxPool2d(kernel_size=self.kernal_size_conv3, stride=self.stride_size_pool3)

    def forward(self, x):
        out = self.conv_block1(x)
        out1 = self.maxPool_block1(out)
       # print("INFO : shape of conv_block_1 output is ", out1.shape)
        out1 = self.conv_block1(out1)
        out2 = self.maxPool_block2(out1)
        # print("INFO : shape of conv_block_2 output is ", out2.shape)
        out2 = self.conv_block3(out2)
        out3 = self.maxPool_block3(out2)
       # print("INFO : shape of conv_block_3 output is ", out3.shape)
        return out3


# combining both vggnet class and dense net class at one place
class BaseModel(nn.Module):

    def __init__(self, input_size, args: ModelArgs) -> None:
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.dnet = DenseNet(input_size=self.input_size, args=args)
        self.vnet = VGGNet(args=args)
        self.flattn = nn.Flatten()

    def forward(self, x):
        out = self.vnet(x)
        out = self.flattn(out)
        out = self.dnet(out)
        return out


def getModelArchitecture():
    vnet = VGGNet(ModelArgs)
    dnet = DenseNet(2187, ModelArgs)
    print("-----------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------")
    print(vnet)
    print()
    print(dnet)
    print("-----------------------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    tensor = torch.randn(size=(16, 3, 224, 224))
    base = BaseModel(input_size=2187, args=ModelArgs)
    out = base(tensor)
    print("output shape is ", out.shape)
    # out = vnet(tensor)
    getModelArchitecture()
