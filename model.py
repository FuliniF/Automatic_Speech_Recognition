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

class LSTMBlock(nn.Module):
    def __init__(self, num_channels):
        super(LSTMBlock, self).__init__()
        self.batch_size = 200
        self.length = 99
        self.input_dim = 2
        self.hidden_dim = 12
        self.num_layers = 1
        self.rearrangev = Rearrange('b c h w -> b (h w) c')

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, 12)

    def forward(self, x):
        input_data = self.rearrangev(x)
        linear_layer = nn.Linear(12,2)
        lin_input = linear_layer(input_data)

        x, _ = self.lstm(lin_input)
        x = self.linear(x[:,-1,:])

        return x

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels*16, (3,1), padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels*16)
        self.conv2 = nn.Conv2d(num_channels*16, num_channels*16, (3,1), padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels*16)
        self.conv3 = nn.Conv2d(num_channels*16, num_channels, (3,1), padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

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

class ValueHead_lstm(nn.Module):
    def __init__(self, length, channel, label_num):
        super(ValueHead_lstm, self).__init__()
        output_channel = 105*channel
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x
    
class model1(nn.Module):
    def __init__(self, length, channel, label_num):
        super(model1, self).__init__()
        self.emb = Embedding()
        '''
        To change network structure:
        change self.net and self.val
        '''
        self.net = ConvBlock(channel)
        # self.net = LSTMBlock(channel)
        self.val = ValueHead(length, channel, label_num)
        # self.val = ValueHead_lstm(length, channel, label_num)

    def forward(self, x):
        # B, 99, 12
        # B, c, h, w
        x = self.emb(x)
        # B, 12, 1, 99
        x = self.net(x)
        # 12 classes
        x = self.val(x)

        return x