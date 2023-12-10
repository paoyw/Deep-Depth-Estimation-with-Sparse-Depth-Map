import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VggEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        else:
            vgg19 = models.vgg19()
        self.layers = nn.ModuleDict(
            {
                'layer0': vgg19.features[:5],
                'layer1': vgg19.features[5:10],
                'layer2': vgg19.features[10:19],
                'layer3': vgg19.features[19:21],
            }
        )

    def encoder_channels(self):
        channels = []
        for layer in ['layer0', 'layer1', 'layer2']:
            channels.append(self.layers[layer][-3].out_channels)
        channels.append(self.layers['layer3'][-2].out_channels)
        return channels

    def forward(self, x):
        features = [x]
        for _, layer in self.layers.items():
            features.append(layer(features[-1]))
        return features[1:]


class Decoder(nn.Module):
    def __init__(self, encoder_channels, scale=80) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = [256, 128, 64, 64]
        self.scale = scale

        self.layers = nn.ModuleDict(
            {
                'layer0': self.__ConvBlock(
                    in_channels=self.encoder_channels[-1],
                    out_channels=self.decoder_channels[0]
                ),
                'layer1': self.__ConvBlock(
                    in_channels=self.encoder_channels[-2] +
                    self.decoder_channels[0],
                    out_channels=self.decoder_channels[1]
                ),
                'layer2': self.__ConvBlock(
                    in_channels=self.encoder_channels[-3] +
                    self.decoder_channels[1],
                    out_channels=self.decoder_channels[2]
                ),
                'layer3': self.__ConvBlock(
                    in_channels=self.encoder_channels[-4] +
                    self.decoder_channels[2],
                    out_channels=self.decoder_channels[3],
                    skip_upsample=True,
                ),
                'conv4': nn.Conv2d(
                    self.decoder_channels[3], 1,
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            }
        )

        nn.init.constant_(self.layers['conv4'].bias, val=15)

    def __ConvBlock(self, in_channels, out_channels, skip_upsample=False):
        if skip_upsample:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
            )

    def forward(self, input_features):
        features = [input_features[-1]]
        features.append(self.layers['layer0'](input_features[-1]))
        features.append(self.layers['layer1'](
            torch.cat((
                features[-1],
                F.interpolate(input_features[-2], size=features[-1].shape[-2:])
            ), dim=1)
        ))
        features.append(self.layers['layer2'](
            torch.cat((
                features[-1],
                F.interpolate(input_features[-3], size=features[-1].shape[-2:])
            ), dim=1)
        ))
        features.append(self.layers['layer3'](
            torch.cat((
                features[-1],
                F.interpolate(input_features[-4], size=features[-1].shape[-2:])
            ), dim=1)
        ))
        features.append(self.layers['conv4'](features[-1]))
        return torch.clip(features[-1], min=-1, max=self.scale)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VggEncoder()
        self.decoder = Decoder(self.encoder.encoder_channels())

    def forward(self, x, resize=True):
        if resize:
            origin_size = x.shape[-2:]

        features = self.encoder(x)
        outputs = self.decoder(features)

        if resize:
            outputs = F.interpolate(outputs, size=origin_size, mode='bilinear')

        return outputs
