import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

torch.manual_seed(970530)
torch.cuda.manual_seed_all(970530)


# This is two parts of the attention module:
## Spatial_Attention in attention module
class spatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)
        spaAtten = q
        spaAtten = torch.squeeze(spaAtten, 1)
        q = self.norm(q)
        # In addition, return to spaAtten for visualization
        return U * q, spaAtten


## Frequency Attention in attention module
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
        self.spatialAttention = spatialAttention(in_channels)

    def forward(self, U):
        U_sse, spaAtten = self.spatialAttention(U)
        U_cse, freqAtten = self.frequencyAttention(U)
        # Return new 4D features
        # and the Frequency Attention and Spatial_Attention
        return U_cse + U_sse, spaAtten, freqAtten


# depthwise separable convolution(DS Conv):
# depthwise conv + pointwise conv + bn + relu
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernal_size = kernel_size
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Context module in DSC module
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3BNReLU, self).__init__()
        self.conv3x3 = depthwise_separable_conv(in_channel, out_channel, 3)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv3x3(x)))


class ContextModule(nn.Module):
    def __init__(self, in_channel):
        super(ContextModule, self).__init__()
        self.stem = Conv3x3BNReLU(in_channel, in_channel // 2)
        self.branch1_conv3x3 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)

    def forward(self, x):
        x = self.stem(x)
        # branch1
        x1 = self.branch1_conv3x3(x)
        # branch2
        x2 = self.branch2_conv3x3_1(x)
        x2 = self.branch2_conv3x3_2(x2)
        # concat
        return torch.cat([x1, x2], dim=1)


# SFT-Net:
# Attention module + DSC module + LSTM module
class SFT_Net(nn.Module):
    def __init__(self, num_classes=1):
        super(SFT_Net, self).__init__()
        self.Atten = sfAttention(in_channels=5)
        self.bneck = nn.Sequential(
            #  begin x = [32, 16, 5, 6, 9], in fact x1 = [32, 5, 6, 9]
            depthwise_separable_conv(5, 32, 3),
            depthwise_separable_conv(32, 64, 3),
            # default dropout
            nn.Dropout2d(0.3),
            depthwise_separable_conv(64, 128, 3),
            # Context Module
            ContextModule(128),
            depthwise_separable_conv(128, 64, 3),
            # default dropout
            nn.Dropout2d(0.3),
            depthwise_separable_conv(64, 32, 3),
            nn.AdaptiveAvgPool2d((2, 2))  # [batch, 32, 2, 2]
        )
        self.linear = nn.Linear(32 * 2 * 2, 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)  # [batch, input_size, -]
        self.linear1 = nn.Linear(32 * 16, 120)
        self.dropout = nn.Dropout(0.4)  # default dropout
        self.linear2 = nn.Linear(120, num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x1 - x16 [batch, 16, 5, 6, 9]
        x1 = torch.squeeze(x[:, 0, :, :, :], 1)  # [batch, 5, 6, 9]
        x2 = torch.squeeze(x[:, 1, :, :, :], 1)
        x3 = torch.squeeze(x[:, 2, :, :, :], 1)
        x4 = torch.squeeze(x[:, 3, :, :, :], 1)
        x5 = torch.squeeze(x[:, 4, :, :, :], 1)
        x6 = torch.squeeze(x[:, 5, :, :, :], 1)
        x7 = torch.squeeze(x[:, 6, :, :, :], 1)
        x8 = torch.squeeze(x[:, 7, :, :, :], 1)
        x9 = torch.squeeze(x[:, 8, :, :, :], 1)
        x10 = torch.squeeze(x[:, 9, :, :, :], 1)
        x11 = torch.squeeze(x[:, 10, :, :, :], 1)
        x12 = torch.squeeze(x[:, 11, :, :, :], 1)
        x13 = torch.squeeze(x[:, 12, :, :, :], 1)
        x14 = torch.squeeze(x[:, 13, :, :, :], 1)
        x15 = torch.squeeze(x[:, 14, :, :, :], 1)
        x16 = torch.squeeze(x[:, 15, :, :, :], 1)
        # spaAtten and freqAtten
        x1, spaAtten1, freqAtten1 = self.Atten(x1)  # [batch, 5, 6, 9], [batch, 6, 9], [batch, 5, 1]
        x2, spaAtten2, freqAtten2 = self.Atten(x2)
        x3, spaAtten3, freqAtten3 = self.Atten(x3)
        x4, spaAtten4, freqAtten4 = self.Atten(x4)
        x5, spaAtten5, freqAtten5 = self.Atten(x5)
        x6, spaAtten6, freqAtten6 = self.Atten(x6)
        x7, spaAtten7, freqAtten7 = self.Atten(x7)
        x8, spaAtten8, freqAtten8 = self.Atten(x8)
        x9, spaAtten9, freqAtten9 = self.Atten(x9)
        x10, spaAtten10, freqAtten10 = self.Atten(x10)
        x11, spaAtten11, freqAtten11 = self.Atten(x11)
        x12, spaAtten12, freqAtten12 = self.Atten(x12)
        x13, spaAtten13, freqAtten13 = self.Atten(x13)
        x14, spaAtten14, freqAtten14 = self.Atten(x14)
        x15, spaAtten15, freqAtten15 = self.Atten(x15)
        x16, spaAtten16, freqAtten16 = self.Atten(x16)
        # attention avg
        spaAtten = (spaAtten1 + spaAtten2 + spaAtten3 + spaAtten4 + spaAtten5 + spaAtten6 + spaAtten7 + spaAtten8
                    + spaAtten9 + spaAtten10 + spaAtten11 + spaAtten12 + spaAtten13 + spaAtten14 + spaAtten15 + spaAtten16) / 16
        freqAtten = (freqAtten1 + freqAtten2 + freqAtten3 + freqAtten4 + freqAtten5 + freqAtten6 + freqAtten7 + freqAtten8
                    -+ freqAtten9 + freqAtten10 + freqAtten11 + freqAtten12 + freqAtten13 + freqAtten14 + freqAtten15 + freqAtten16) / 16
        # bneck
        x1 = self.bneck(x1)
        x2 = self.bneck(x2)
        x3 = self.bneck(x3)
        x4 = self.bneck(x4)
        x5 = self.bneck(x5)
        x6 = self.bneck(x6)
        x7 = self.bneck(x7)
        x8 = self.bneck(x8)
        x9 = self.bneck(x9)
        x10 = self.bneck(x10)
        x11 = self.bneck(x11)
        x12 = self.bneck(x12)
        x13 = self.bneck(x13)
        x14 = self.bneck(x14)
        x15 = self.bneck(x15)
        x16 = self.bneck(x16)

        x1 = self.linear(x1.view(x1.shape[0], 1, -1))  # [batch, 1, 32*2*2] -> [batch, 1, 64]
        x2 = self.linear(x2.view(x2.shape[0], 1, -1))
        x3 = self.linear(x3.view(x3.shape[0], 1, -1))
        x4 = self.linear(x4.view(x4.shape[0], 1, -1))
        x5 = self.linear(x5.view(x5.shape[0], 1, -1))
        x6 = self.linear(x6.view(x6.shape[0], 1, -1))
        x7 = self.linear(x7.view(x7.shape[0], 1, -1))
        x8 = self.linear(x8.view(x8.shape[0], 1, -1))
        x9 = self.linear(x9.view(x9.shape[0], 1, -1))
        x10 = self.linear(x10.view(x10.shape[0], 1, -1))
        x11 = self.linear(x11.view(x11.shape[0], 1, -1))
        x12 = self.linear(x12.view(x12.shape[0], 1, -1))
        x13 = self.linear(x13.view(x13.shape[0], 1, -1))
        x14 = self.linear(x14.view(x14.shape[0], 1, -1))
        x15 = self.linear(x15.view(x15.shape[0], 1, -1))
        x16 = self.linear(x16.view(x16.shape[0], 1, -1))

        # cat 16 * [batch, 1, 32] -> [batch, 16, 32]
        out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1)

        # after LSTM                    [batch, 16, 64]
        out, (h, c) = self.lstm(out)

        # flatten                       [batch, 16*120]
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])

        # first linear                  [batch, 120]
        out = self.linear1(out)
        out = self.dropout(out)
        # second linear                 [batch, ]
        out = self.linear2(out)
        # finnal feature and attention  [batch, 1]
        return out, spaAtten, freqAtten


if __name__ == '__main__':
    # input = torch.rand((32, 16, 5, 6, 9))
    input = torch.rand((32, 10, 25, 63))
    net = SFT_Net()
    output, spaAtten, freqAtten = net(input)
    print("Input shape     : ", input.shape)
    print("Output shape    : ", output.shape)
    print("spaAtten shape  : ", spaAtten.shape)
    print("freqAtten shape : ", freqAtten.shape)