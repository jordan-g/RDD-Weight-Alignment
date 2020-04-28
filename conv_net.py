import torch
import torch.nn as nn

from layers import LinearFA, Conv2dFA

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ConvNet(nn.Module):
    def __init__(self, input_channels, use_backprop=False):
        super(ConvNet, self).__init__()

        self.feature_layers = []

        Linear = LinearFA if (not use_backprop) else nn.Linear
        Conv2d = Conv2dFA if (not use_backprop) else nn.Conv2d

        self.conv1 = Conv2d(input_channels, 64, bias=False, kernel_size=5)
        self.feature_layers.append(self.conv1)
        self.feature_layers.append(nn.ReLU(inplace=True))

        self.feature_layers.append(nn.BatchNorm2d(64))

        self.feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = Conv2d(64, 64, bias=False, kernel_size=5)
        self.feature_layers.append(self.conv2)
        self.feature_layers.append(nn.ReLU(inplace=True))

        self.feature_layers.append(nn.BatchNorm2d(64))

        self.feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        self.fc1 = Linear(1600, 384)
        self.classification_layers.append(self.fc1)
        self.classification_layers.append(nn.ReLU(inplace=True))

        self.classification_layers.append(nn.BatchNorm1d(384))

        self.fc2 = Linear(384, 192)
        self.classification_layers.append(self.fc2)
        self.classification_layers.append(nn.ReLU(inplace=True))

        self.classification_layers.append(nn.BatchNorm1d(192))

        self.fc3 = Linear(192, 10)
        self.classification_layers.append(self.fc3)
        self.classification_layers.append(nn.Softmax(dim=-1))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

    def forward(self, x):
        return self.out(x)

    def decay_fb_weights(self, weight_decay):
        self.conv1.decay_fb_weights(weight_decay)
        self.conv2.decay_fb_weights(weight_decay)
        self.fc1.decay_fb_weights(weight_decay)
        self.fc2.decay_fb_weights(weight_decay)
        self.fc3.decay_fb_weights(weight_decay)