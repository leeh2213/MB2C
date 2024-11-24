import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from torchvision.models import resnet50
from model.EEGModel import *
import clip

class ImageResNet(nn.Module):

    def __init__(self, projection_dim=128):
        super(ImageResNet, self).__init__()
        self.out        = projection_dim
        # self.encoder    = resnet18(pretrained=True)
        self.encoder    = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.encoder.fc = nn.Sequential(    
                                            nn.Linear(self.encoder.fc.in_features, self.encoder.fc.out_features),
                                            nn.LeakyReLU(),
                                            nn.Dropout(p=0.05),
                                            nn.Linear(self.encoder.fc.out_features, projection_dim, bias=False)
                                        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, dim=-1)
        return x

class ImageCLIPNet(nn.Module):

    def __init__(self, projection_dim=128, device= "cuda"):
        super(ImageCLIPNet, self).__init__()
        self.out        = projection_dim
        # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        self.encoder, preprocess = clip.load('ViT-L/14', device, jit=False)
        self.preprocess = preprocess
        for params in self.encoder.parameters():
            params.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(768, projection_dim, bias=False)
                                        )

    def forward(self, x):
        x = self.encoder.encode_image(x)
        x = x.to(torch.float32) 
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        return x

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1000, out_dim=1000):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=1000, hidden_dim=512, out_dim=1000): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
        # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        # model.fc = nn.Linear(
        #     in_features=2048, out_features=args.embedding_size)
        # # 冻结所有参数
        # for param in model.parameters():
        #     param.requires_grad = False

        # # 解冻 nn.Linear 的参数
        # for param in model.fc.parameters():
        #     param.requires_grad = True


class Embedder(nn.Module):
    def __init__(self, image_backbone=resnet50(),eeg_backbone=CompactEEGNet()):
        super().__init__()
        
        self.image_backbone = image_backbone
        self.eeg_backbone = eeg_backbone
        self.projector = projection_MLP(image_backbone.output_dim)

        self.image_encoder = nn.Sequential( # f encoder
            self.image_backbone,
            self.projector
        )
        self.eeg_encoder = nn.Sequential( # f encoder
            self.eeg_backbone,
            # self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, image, eeg):
        f_i,f_e,  h = self.image_encoder, self.eeg_encoder,self.predictor
        # 512, 2048
        image,eeg = f_i(image),f_e(eeg)
        # 512, 2048
        p1, p2 = h(image),h(eeg)
        
        return p1, p2

def get_backbone(backbone,embeddig_size, castrate=True):
    backbone = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    backbone.output_dim = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    # 冻结所有参数
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone

if __name__ == "__main__":
    model =  Embedder(get_backbone('renet50',1000))
    x1 = torch.randn((32, 3, 224, 224))

    out=model.forward(x1).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)














