import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.rearrangev = Rearrange('b (h w) c -> b c h w', h=1)

    def forward(self, x):
        x = self.rearrangev(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels*16, (3,1), padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels*16)
        self.conv2 = nn.Conv2d(num_channels*16, num_channels*16, (3,1), padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels*16)
        self.conv3 = nn.Conv2d(num_channels*16, num_channels, (3,1), padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        print("Residual Block")

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.relu(x)
        return x

class ValueHead(nn.Module):
    def __init__(self, length, channel, label_num):
        super(ValueHead, self).__init__()
        self.rearrangev = Rearrange('b c h w -> b (c h w)')
        output_channel = 105*channel
        self.fc1 = nn.Linear(105*channel, output_channel)
        self.fc2 = nn.Linear(output_channel, label_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.rearrangev(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    
class fcmodel1(nn.Module):
    def __init__(self, length, channel, label_num):
        super(fcmodel1, self).__init__()
        self.emb = Embedding()
        self.net = ResidualBlock(channel)
        self.val = ValueHead(length, channel, label_num)

    def forward(self, x):
        x = self.emb(x)
        x = self.net(x)
        x = self.val(x)
        
        return x


# inp = torch.rand(1, 99, 12)

# net = ResidualBlock(12)
# emb = Embedding()
# val = ValueHead(105, 12, 12)

# network = fcmodel1(99, 12, 12)

# print(network(inp))
# inp = emb(inp)
# print(inp.shape)
# inp = net(inp)
# print(inp.shape)
# inp = val(inp)
# print(inp.shape)