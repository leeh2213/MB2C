import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

class PowerLayer(nn.Module):
    '''
    The power layer: calculates the log-transformed power of the data
    '''
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))
    
class TSception(nn.Module):
    # Self
    def conv_block(self, in_chan, out_chan, kernel, pool, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=(1,1)),
            nn.ReLU())
    # TSception 原始
    # def conv_block(self, in_chan, out_chan, kernel, pool, stride):
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
    #                   kernel_size=kernel, stride=(1,1)),
    #         # nn.AvgPool2d(kernel_size=(1, pool), stride=(1, stride)),
    #         # nn.BatchNorm2d(out_chan),
    #         # nn.ELU()
    #         )
    # def conv_block(self, in_chan, out_chan, kernel, pool, stride):
    #     return nn.Sequential(
    #         nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
    #         PowerLayer(dim=-1, length=pool, step=stride)
    #     )
    
    def __init__(self, sampling_rate=250, num_T=15, num_S=40, shared_ratio=0.4, emb_size=25):
        super().__init__()
        self.inception_window = [0.5, 0.2, 0.1]
        self.pool = 51
        self.pool_step_rate = 0.1
        self.stride=int(self.pool_step_rate*self.pool)

        self.Tception1 = self.conv_block(1, emb_size, (1, int(self.inception_window[0] * sampling_rate)), self.pool, self.stride)
        self.Tception2 = self.conv_block(1, emb_size, (1, int(self.inception_window[1] * sampling_rate)), self.pool, self.stride)
        self.Tception3 = self.conv_block(1, emb_size, (1, int(self.inception_window[2] * sampling_rate)), self.pool, self.stride)
        # revised from shallownet
        self.conv1=nn.Conv2d(1, 40, (1, 125), (1, 1))
        self.conv2=nn.Conv2d(1, 40, (1, 50), (1, 1))
        self.conv3=nn.Conv2d(1, 40, (1, 25), (1, 1))
        self.pool1=nn.AvgPool2d((1, 51), (1, 5))   
        self.pool2=nn.AvgPool2d((1, 51), (1, 5))  
        self.pool3=nn.AvgPool2d((1, 51), (1, 5))                                                                                                                                                                                       
        self.tsconv = nn.Sequential(
            # nn.Conv2d(1, 40, (1, 25), (1, 1)),
            # nn.AvgPool2d((1, 51), (1, 5)),
            # nn.BatchNorm2d(40),
            # nn.ELU(),
            nn.Conv2d(emb_size, emb_size, (63, 1), (1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.BN_t = nn.BatchNorm2d(emb_size)
        self.BN_t_ = nn.BatchNorm2d(emb_size)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        self.depthwise_conv = nn.Conv2d(emb_size, 1*emb_size, (63, 1), groups=emb_size)
        self.avgpool2 = nn.AvgPool2d((1,8))
        self.dropout = nn.Dropout(p=0.5)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1,4))

    def forward(self, x: Tensor) -> Tensor:
        """
        EEG Tception
        """
        x_ca = x
        y = self.Tception1(x_ca)
        out = y
        y = self.Tception2(x_ca)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x_ca)
        # 
        out = torch.cat((out, y), dim=-1)
        # b, _, _, _ = x.shape
        out = self.BN_t(out)
        out = self.depthwise_conv(out)
        out = self.elu(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        # out = self.OneXOneConv(out)
        # out = self.BN_t_(out)
        # x = out
        # x = self.tsconv(x)
        x = self.projection(x)
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
        # self.separable_conv = SeparableConv2d(
        #     in_channels=d*f1, 
        #     out_channels=f2, 
        #     kernel_size=(1,16)
        # )
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


if __name__ == "__main__":
    model = TSception()
    x1 = torch.randn((1000, 1, 63, 250))

    model.forward(x1).backward()
    print("forward backwork check")












