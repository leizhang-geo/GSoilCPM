# coding=utf-8

import torch
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg


class ConvNet(nn.Module):
    def __init__(self, num_channels_band, num_channels_topo, num_channels_climate, num_channels_vege, num_channels_bedrock):
        super(ConvNet, self).__init__()
        self.num_channels_band = num_channels_band
        self.num_channels_topo = num_channels_topo
        self.num_channels_climate = num_channels_climate
        self.num_channels_vege = num_channels_vege
        self.num_channels_bedrock = num_channels_bedrock

        self.cnn_out_channels = 32

        self.features_band = nn.Sequential(
            nn.Conv2d(self.num_channels_band, self.cnn_out_channels, kernel_size=1, stride=1, padding=0),
        )

        # self.features_topo = nn.Sequential(
        #     nn.Conv2d(self.num_channels_topo, 32, kernel_size=2, stride=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=1),
        #     nn.Conv2d(32, self.cnn_out_channels, kernel_size=2, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.features_topo = nn.Sequential(
            nn.Conv2d(self.num_channels_topo, self.cnn_out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.features_climate = nn.Sequential(
            nn.Conv2d(self.num_channels_climate, self.cnn_out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.features_vege = nn.Sequential(
            nn.Conv2d(self.num_channels_vege, self.cnn_out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.features_bedrock = nn.Sequential(
            nn.Conv2d(self.num_channels_bedrock, self.cnn_out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.features_merge = nn.Sequential(
            nn.Conv2d(self.cnn_out_channels*5, 128, kernel_size=1, stride=1, padding=0),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.Linear(256, 64),
        #     nn.Linear(64, cfg.num_class),
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(128, cfg.num_class),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        # self.final_classifier = nn.Sequential(
        #     nn.Linear(128, cfg.num_class),
        # )

        self.final_regressor = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x_band, x_topo, x_climate, x_vege, x_bedrock):
        x_band = self.features_band(x_band)
        x_topo = self.features_topo(x_topo)
        x_climate = self.features_climate(x_climate)
        x_vege = self.features_vege(x_vege)
        x_bedrock = self.features_bedrock(x_bedrock)
        # print('x_topo: ', x_topo.size())
        # print('x_climate: ', x_climate.size())
        x = torch.concatenate((x_band, x_topo, x_climate, x_vege, x_bedrock), dim=1)
        # print(x.size())
        x = self.features_merge(x)
        # print(x.size())
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        x = self.classifier(x)
        # x = self.final_classifier(x)
        x = self.final_regressor(x)
        x = x.reshape(-1)
        return x

    def num_flat_features(self, x):
        sizes = x.size()[1:]
        num_features = 1
        for s in sizes:
            num_features *= s
        return num_features
