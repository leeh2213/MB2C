import torch.nn as nn
import torch
# This is two parts of the attention module:
## Spatial_Attention in attention module
class spatialAttention(nn.Module):
    def __init__(self, data_length):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(data_length, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        # 256, 25, 1, 63
        q = self.Conv1x1(U)
        spaAtten = q
        spaAtten = torch.squeeze(spaAtten, 1)
        q = self.norm(q)
        # In addition, return to spaAtten for visualization
        return U * q, spaAtten

class frequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2,
                                      kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels,
                                         kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)
        z = self.Conv_Squeeze(z)
        z = self.Conv_Excitation(z)
        freqAtten = z
        freqAtten = torch.squeeze(freqAtten, 3)
        z = self.norm(z)
        # In addition, return to freqAtten for visualization
        return U * z.expand_as(U), freqAtten


# Attention module:
# spatial-frequency attention
class sfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.frequencyAttention = frequencyAttention(in_channels)
        # self.spatialAttention = spatialAttention(in_channels)

    def forward(self, U):
        # U_sse, spaAtten = self.spatialAttention(U)
        U_cse, freqAtten = self.frequencyAttention(U)
        # Return new 4D features
        # and the Frequency Attention and Spatial_Attention
        # return U_cse + U_sse, spaAtten, freqAtten
        return U_cse, freqAtten



class channel_wise_attention(nn.Module):
    def __init__(self, H, W, C, reduce):
        super(channel_wise_attention, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.r = reduce
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(self.C, self.r),
            nn.Tanh(),
            nn.Linear(self.r, self.C)
        )
        # softmax
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        # mean pooling
        # x:256, 1, 250, 63; x1:256, 63, 1, 250
        # x1 = x.permute(0, 3, 1, 2)
        x1 = x
        x = x.permute(0, 1, 3, 2)
        mean = nn.AvgPool2d((1, x.shape[2]))
        # feature_map: 256, 1, 1, 63
        feature_map = mean(x1).permute(0, 2, 3, 1)
        # 256, 1, 1, 63
        feature_map_fc = self.fc(feature_map)
        # softmax
        v = self.softmax(feature_map_fc)
        # channel_wise_attention
        # 10,32
        v = v.reshape(-1, self.C)
        # vr: 256, 1, 250, 63
        vr = torch.reshape(torch.cat([v] * (self.H * self.W), axis=1), [-1, self.H, self.W, self.C])
        # 256, 1, 250, 63
        channel_wise_attention_fm = x * vr
        # v:10,32
        # channel_wise_attention_fm:10,1,284,32
        return v, channel_wise_attention_fm

if __name__ == '__main__':
    x=torch.rand(256, 10, 63, 25)
    # model=CompactEEGNet()
    model=channel_wise_attention(1, 250, 63, 15)
    x_map1, x=model(x)
    print()