import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.Attention import *
from einops import rearrange, reduce, repeat
from torch import Tensor

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=1, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
            
        if isinstance(kernel_size, tuple):
            padding = (
                kernel_size[0]//2 if kernel_size[0]-1 != 0 else 0,
                kernel_size[1]//2 if kernel_size[1]-1 != 0 else 0
            )
            
        self.depthwise = DepthwiseConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CompactEEGNet(nn.Module):
    """
    EEGNet: Compact Convolutional Neural Network (Compact-CNN)
    Compact Convolutional Neural Networks for Classification of Asynchronous Steady-state Visual Evoked Potentials
    https://arxiv.org/pdf/1803.04566.pdf
    """
    '''
    signal_length: length of temporal convolution in first layer
    '''
    def __init__(self, num_channel=63, num_classes=1000, signal_length=125, f1=14, f2=154, d=11):
        super().__init__()
        
        self.signal_length = signal_length
        # layer 1
        self.conv1 = nn.Conv2d(1, f1, (1, signal_length), padding=(0,signal_length//2))
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv = nn.Conv2d(f1, d*f1, (num_channel, 1), groups=f1)
        self.bn2 = nn.BatchNorm2d(d*f1)
        self.avgpool1 = nn.AvgPool2d((1,4))
        # layer 2
        self.separable_conv = SeparableConv2d(
            in_channels=d*f1, 
            out_channels=f2, 
            kernel_size=(1,16)
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.avgpool2 = nn.AvgPool2d((1,8))
        self.dropout = nn.Dropout(p=0.5)
        self.elu = nn.ELU()
        # layer 3
        # self.fc = nn.Linear(in_features=f2*(signal_length//32), out_features=num_classes)
        self.fc = nn.Linear(in_features=1078, out_features=num_classes)

    # In: (B, 1, Chans, Samples, 1) x: B, Chans, Samples
    # Out: (B, embedding)
    def forward(self, x):
        # layer 1
        x = torch.unsqueeze(x,1) # B, Chans, Samples -> B, 1, Chans, Samples
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout(x)
                
        # layer 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)
                
        # layer 3
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.normalize(x ,dim=1)
        
        return x

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
        self.Tception1 = self.conv_block(depth, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(depth, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(depth, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)
        # other
        self.Tception5 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * 256)), 1, self.pool)
        self.Tception6 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * 256)), 1, self.pool)
        self.Tception7 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * 256)), 1, self.pool)

        self.Sception = self.conv_block(num_T, num_S, (num_channel, 1), 1, int(self.pool * 0.25))

        """
         LMDA-Net
        """
        """
        C: num of channels
        W: num of time samples
        k: learnable kernel size
        """
        # self.depthAttention = EEGDepthAttention(W, C, k=7)1:52 4:277
        self.depthAttention = EEGDepthAttention(66, num_channel, k=9)

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

    # x:128,1,10,1024
    def forward(self, x):
        x = x.unsqueeze(1)
        if self.is_channel_wise:
            x = x.permute(0, 1, 3, 2)
            x_map1, x = self.channel_wise_attention(x)
            x = x.permute(0, 1, 3, 2)
            # (256, 1, 63, 250)->(256, 9, 63, 250)
            x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        """
        EEG Tception
        """
        y = self.Tception1(x) # 256, 15, 63, 126
        out = y
        y = self.Tception2(x) # 256, 15, 63, 189
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x) # 256, 15, 63, 220
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
        out = self.depthAttention(out) #256, 15, 63, 66
        """
        Sception
        """
        # B,W,C,H
        out = self.Sception(out)
        out = self.BN_s(out)
        # 256, 15, 1, 66
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        out = self.classifier(out, self.fixed_weight.to(out.device))
        out = F.normalize(out ,dim=1)

        return out

class TimeAttentionNet(nn.Module):
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
        super(TimeAttentionNet, self).__init__()
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

        self.Sception = self.conv_block(num_T, num_S, (num_channel, 1), 1, int(self.pool * 0.25))

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
        self.Atten = sfAttention(in_channels=10)

    # x:128,1,10,1024
    def forward(self, x):
        """
        Sception
        """   
        # x = x.reshape(-1, 63, 10, 25)
        # x = x.permute(0, 2, 1, 3) # 256, 10, 63, 25
        # x, freqAtten = self.Atten(x)
        # # x = self.depthAttention(x)
        # x = x.permute(0,2,1,3) # 256, 63, 10, 25
        # b,c,_,_ =x.shape
        # x = x.reshape(b,c,-1)
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
        # y = self.Tception1(x) # 256, 15, 63, 126
        # out = y
        # y = self.Tception2(x) # 256, 15, 63, 189
        # out = torch.cat((out, y), dim=-1)
        # y = self.Tception3(x) # 256, 15, 63, 220
        # out = torch.cat((out, y), dim=-1) # 256, 15, 63, 535
        y = self.Tception5(x) # 256, 15, 63, 126
        out = y
        y = self.Tception6(x) # 256, 15, 63, 189
        out = torch.cat((out, y), dim=-1)
        y = self.Tception7(x) # 256, 15, 63, 220
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
        out = self.Sception(out)
        out = self.BN_s(out)

        # 256, 15, 1, 66
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # # out = self.classifier(out, self.fixed_weight.to(out.device))
        # out = F.normalize(out ,dim=1)

        return out

class Transformer_Attention(nn.Sequential):
    def __init__(self, emb_size=40, depth=3):
        super().__init__(
            Transformer(depth, emb_size)
        )

class Transformer(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerBlock(emb_size) for _ in range(depth)])

class TransformerBlock(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__(
            TransformerEncoderBlock(emb_size),
            TransformerDecoderBlock(emb_size)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerDecoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd_Dec1(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention_Dec(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd_Dec2(nn.Sequential(
                MultiHeadAttention_Enc_Dec(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd_Dec1(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = y
        y = self.fn(y, **kwargs)
        y += res
        return (x, y)
       
class ResidualAdd_Dec1(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return (x, y)

class ResidualAdd_Dec2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

        self.lm = nn.LayerNorm(40)

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.lm(x)
        x = self.fn((x, y), **kwargs)
        x += res
        return (x, y)



class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class MultiHeadAttention_Enc_Dec(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, X, mask: Tensor = None) -> Tensor:
        x_enc, x_dec = X[0], X[1] # enc is target, dec is source
        queries = rearrange(self.queries(x_dec), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x_dec), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x_enc), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class MultiHeadAttention_Dec(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x = X[1] # target data
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


# if __name__ == '__main__':
#     x=torch.rand(256, 63, 250)
#     # model=CompactEEGNet()
#     model=TimeAttentionNet()
#     output=model(x)
#     print()