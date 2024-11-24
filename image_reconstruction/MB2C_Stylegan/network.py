import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import config
from torchvision.models import googlenet
from einops.layers.torch import Rearrange

np.random.seed(45)
torch.manual_seed(45)

class EEGFeatNet(nn.Module):
    def __init__(self, n_classes=40, in_channels=128, n_features=128, projection_dim=128, num_layers=1):
        super(EEGFeatNet, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        # self.embedding  = nn.Embedding(num_embeddings=in_channels, embedding_dim=n_features)
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc         = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)

    def forward(self, x):

        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device) 
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device)

        _, (h_n, c_n) = self.encoder( x, (h_n, c_n) )

        feat = h_n[-1]
        x = self.fc(feat)
        x = F.normalize(x, dim=-1)
        return x


class ImageFeatNet(nn.Module):

    def __init__(self, projection_dim=128):
        super(ImageFeatNet, self).__init__()
        self.out        = projection_dim
        self.encoder    = googlenet(pretrained=True)
        for params in self.encoder.parameters():
            params.requires_grad = False

        self.base_conv  = nn.Sequential(
                                            nn.Conv2d(in_channels=config.n_subjects+3, out_channels=3, kernel_size=3, padding=1),
                                            nn.ReLU()
                                        )

        self.encoder.fc = nn.Sequential(    
                                            nn.Linear(self.encoder.fc.in_features, 1024),
                                            nn.LeakyReLU(),
                                            nn.Dropout(p=0.05),
                                            nn.Linear(1024, projection_dim, bias=False)
                                        )

    def forward(self, x):
        x = self.base_conv(x)
        x = self.encoder(x)
        x = F.normalize(x, dim=-1)
        return x


class EEGCNNFeatNet(nn.Module):
    def __init__(self, input_shape=(1, 440, 128), n_features=128, projection_dim=128, num_filters=[128, 256, 512, 1024], kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2], padding=[1, 1, 1, 1]):
        super(EEGCNNFeatNet, self).__init__()

        # Define the convolutional layers
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]
        for i, out_channels in enumerate(num_filters):
            self.layers.append( 
                                nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_sizes[i], strides[i], padding[i], bias=False),\
                                    # nn.BatchNorm2d(out_channels),\
                                    nn.InstanceNorm2d(out_channels),\
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.05),
                                )
                             )
            in_channels = out_channels

        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], projection_dim, bias=False),
        )

    def forward(self, x):
        # Apply the convolutional layers
        for layer in self.layers:
            x = layer(x)

        # Flatten the output
        x = torch.reshape(x, [x.shape[0], -1])

        # Apply the fully connected layers
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        # x    = self.proj_layers(feat)
        return x

    def get_cnn_feat_out(self, x):
    	# Apply the convolutional layers
        for layer in self.layers:
            x = layer(x)

        # Flatten the output
        x = x.view(x.shape[0], -1)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)

    def forward(self, x):
        projected = self.projection(x)
        return projected

class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1160, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class CLIPModel(torch.nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super(CLIPModel, self).__init__()
        self.text_encoder     = text_encoder
        self.image_encoder    = image_encoder
        self.fc = nn.Identity()

    
    def forward(self, texts, images):
        text_feat  = self.text_encoder(texts)
        image_feat = self.image_encoder(images)
        # text_embed  = torch.nn.functional.normalize( text_feat, dim=-1 )
        # image_embed = torch.nn.functional.normalize( image_feat, dim=-1 )
        text_feat = self.fc(text_feat)

        return text_feat, image_feat



class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 50), (1, 1)),
            nn.AvgPool2d((1, 102), (1, 10)),
            nn.BatchNorm2d(40),
            nn.ELU(),
        )

        self.sconv = nn.Sequential(
            nn.Conv2d(40, 40, (128, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x [:,np.newaxis,:,:]
        x = self.tconv(x)
        x = self.sconv(x)
        x = self.projection(x)
        return x
      
class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead(),
            Proj_eeg()
        )
