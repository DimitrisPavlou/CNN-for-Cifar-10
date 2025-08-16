import torch
import torchvision
from torch import nn
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader , random_split
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module) :

    def __init__(self , in_channels , hidden_units1 , hidden_units2, output_shape) :

        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels , out_channels = hidden_units1 , kernel_size = 3 , stride = 1 , padding = "same") ,
            nn.ReLU() ,
            nn.BatchNorm2d(hidden_units1),
            nn.Conv2d(in_channels = hidden_units1 , out_channels = hidden_units1 , kernel_size = 3 , stride = 1 , padding = "same") ,
            nn.ReLU() ,
            nn.BatchNorm2d(hidden_units1),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units1 , out_channels = hidden_units2 , kernel_size = 3 , stride = 1 , padding = "same") ,
            nn.ReLU() ,
            nn.BatchNorm2d(hidden_units2),
            nn.Conv2d(in_channels = hidden_units2 , out_channels = hidden_units2 , kernel_size = 3 , stride = 1 , padding = "same") ,
            nn.BatchNorm2d(hidden_units2),
            nn.ReLU() ,
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units2 , out_channels = hidden_units1 , kernel_size = 3 , stride = 1 , padding = "same") ,
            nn.ReLU() ,
            nn.BatchNorm2d(hidden_units1),
            nn.Conv2d(in_channels = hidden_units1 , out_channels = hidden_units1 , kernel_size = 3 , stride = 1 , padding = "same") ,
            nn.ReLU() ,
            nn.BatchNorm2d(hidden_units1),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten() ,
            nn.Linear(in_features = hidden_units1*4*4 , out_features = 10) ,
        )

    def forward(self , x) :
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)

