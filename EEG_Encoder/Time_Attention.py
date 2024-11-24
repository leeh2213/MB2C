import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from Attention import *


class WeightFreezing(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, multiple=0):
        super(WeightFreezing, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        mask = torch.rand(input_dim, output_dim) < shared_ratio
        self.register_buffer('shared_mask', mask)
        self.register_buffer('independent_mask', ~mask)

        self.multiple = multiple

    def forward(self, x, shared_weight):
        combined_weight = torch.where(self.shared_mask, shared_weight * self.multiple, self.weight.t())
        output = F.linear(x, combined_weight.t(), self.bias)
        return output

class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """

    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        # print('查看参数是否变化:', conv.bias)

        return y * self.C * x
  
class TSANet(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))
        )

    def __init__(self, num_channel=63, num_classes=1000, sampling_rate = 250, num_T = 15, num_S = 15, 
                 is_channel_wise = True, shared_ratio=0.4):
        # input_size: 1 x EEG channel x datapoint
        super(TSANet, self).__init__()
        # self.inception_window = [0.5, 0.25, 0.125, 0.0625]
        self.inception_window = [0.5, 0.25, 0.125]
        # self.inception_window = [0.25, 0.125, 0.0625]
        self.pool = 8
        self.is_channel_wise = is_channel_wise
        self.num_classes=num_classes

        # self.reduce = 5
        # best
        self.reduce = 15
        # best
        # depth = 2
        # depth = 9 90.23
        # depth = 7 0.8571
        depth = 9

        '''
        by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        achieve the 1d convolution operation
        '''
        self.Tception1 = self.conv_block(num_T, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(num_T, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(num_T, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)
        # other
        self.Tception5 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * 256)), 1, self.pool)
        self.Tception6 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * 256)), 1, self.pool)
        self.Tception7 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * 256)), 1, self.pool)

        self.Sception = self.conv_block(1, num_S, (num_channel, 1), 1, int(self.pool * 0.25))

        """
         LMDA-Net
        """
        """
        C: num of channels
        W: num of time samples
        k: learnable kernel size
        """
        # self.depthAttention = EEGDepthAttention(W, C, k=7)1:52 4:277
        self.depthAttention = EEGDepthAttention(25, num_channel, k=9)

        self.channel_weight = nn.Parameter(torch.randn(depth, 1, num_channel), requires_grad=True)
        '''end'''

        'Weight-Freezing'
        self.classifier = WeightFreezing(990 , num_classes, shared_ratio=shared_ratio)
        self.shared_weights = nn.Parameter(torch.Tensor(num_classes, 990), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask
        'end'

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        # self.H = 1,self.W = sampling_rate
        self.channel_wise_attention = channel_wise_attention(1, sampling_rate, num_channel, self.reduce)
        self.pool_time = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8))

        self.fc = nn.Linear(in_features=990, out_features=num_classes)
        # in_channels：分为多少个时间段
        self.Atten = sfAttention(in_channels=25)

    # x:128,1,10,1024
    def forward(self, x):
        """
        Sception
        """   
        x = self.Sception(x)
        x = self.BN_s(x)
        x = x.reshape(-1, 15, 10, 25)
        x = x.permute(0, 2, 1, 3) # 256, 10, 63, 25
        x, freqAtten = self.Atten(x)
        # x = self.depthAttention(x)
        x = x.permute(0,2,1,3) # 256, 63, 10, 25
        b,c,_,_ =x.shape
        x = x.reshape(b,c,-1)
        x = x.unsqueeze(1)
        # if self.is_channel_wise:
        #     # 256, 1, 250, 63
        #     x = x.permute(0, 1, 3, 2)
        #     x_map1, x = self.channel_wise_attention(x)
        #     x = x.permute(0, 1, 3, 2)
        #     # (256, 1, 63, 250)->(256, 9, 63, 250)
        #     x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        """
        EEG Tception
        """
        y = self.Tception1(x) # 256, 15, 63, 126
        out = y
        y = self.Tception2(x) # 256, 15, 63, 189
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x) # 256, 15, 63, 220
        out = torch.cat((out, y), dim=-1) # 256, 15, 63, 535
        # y = self.Tception5(x) # 256, 15, 63, 126
        # out = y
        # y = self.Tception6(x) # 256, 15, 63, 189
        # out = torch.cat((out, y), dim=-1)
        # y = self.Tception7(x) # 256, 15, 63, 220
        out = torch.cat((out, y), dim=-1) # 256, 15, 63, 535
        """
        reduceLayer
        """
        # 256, 15, 63, 535 -> 256, 15, 63, 66
        out = self.pool_time(out)
        out = self.BN_t(out)
        """
        depthAttention
        """ 
        # out = self.depthAttention(out) #256, 15, 63, 66


        # 256, 15, 1, 66
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        out = self.classifier(out, self.fixed_weight.to(out.device))
        out = F.normalize(out ,dim=1)

        return out


if __name__ == '__main__':
    x=torch.rand(256, 63, 250)

    # model=CompactEEGNet()
    model=TSANet()
    output=model(x)
    print()