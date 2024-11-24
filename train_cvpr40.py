## Take input of EEG and save it as a numpy array
from configs import cvpr40config as config
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from encoder_modules import *
from util.CLIPModel import CLIPModel
from Loss.ganModelCls import _param,_netG,_netD,_netG2,_netD2
from torch.autograd import Variable
from util.augment import augment_fuc
import torch.autograd as autograd

args = config.args
base_path       = args.base_path
train_path      = args.train_path
validation_path = args.validation_path
device          = args.device
print(device)

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.0)

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * GP_LAMBDA
    return gradient_penalty

def reset_grad(nets):
    for net in nets:
        net.zero_grad()

def Cycle_GAN(image_features, eeg_features, labels, labels_a, labels_b, lam, temperature):
    
    with torch.autograd.set_detect_anomaly(True):

        """ Discriminator """
        for _ in range(5):
            eeg_feat = eeg_features
            X = Variable(image_features)
            z = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            y_true = Variable(labels).to(device)

            D_real, C_real = netD(X) 
            D_loss_real = torch.mean(D_real) 
            if is_aug:
                C_loss_real = F.cross_entropy(C_real[:batch_size], y_true) + lam * F.cross_entropy(C_real[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_real[batch_size:], labels_b)
            else:
                C_loss_real = F.cross_entropy(C_real, y_true)
                
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward(retain_graph=True)
            
            G_sample, _ = netG(z, eeg_feat)
            
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            
            if is_aug:
                C_loss_fake = F.cross_entropy(C_fake[:batch_size], y_true) + lam * F.cross_entropy(C_fake[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_fake[batch_size:], labels_b)
            else:
                C_loss_fake = F.cross_entropy(C_fake, y_true)
                
            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward(retain_graph=True)

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward(retain_graph=True)

            # Wasserstein Loss
            Wasserstein_D = D_loss_real - D_loss_fake
            optimizersCAN['optimizerD'].step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            eeg_feat = Variable(eeg_features)
            X = Variable(image_features)
            z = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            y_true = Variable(labels).to(device)

            G_sample, _ = netG(z, eeg_feat)
            D_fake, C_fake = netD(G_sample)
            _, C_real = netD(X)
            # GAN's G loss
            G_loss = torch.mean(D_fake)
            if is_aug:
                C_loss = (F.cross_entropy(C_real[:batch_size], y_true) + F.cross_entropy(C_fake[:batch_size], y_true)) / 2 + (lam * F.cross_entropy(C_real[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_real[batch_size:], labels_b) + lam * F.cross_entropy(C_fake[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_fake[batch_size:], labels_b)) / 2
            else:
                C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2
            
            GC_loss = -G_loss + C_loss
            
            reg_loss = Variable(torch.Tensor([0.0])).to(device)
            if REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(REG_W_LAMBDA)
                
            all_loss = GC_loss + reg_loss
            all_loss.backward(retain_graph=True)
            optimizersCAN['optimizerG'].step()
            reset_grad(nets)
            
        """Discriminator2"""
        for _ in range(5):
            eeg_feat = Variable(eeg_features)
            X = Variable(image_features)
            z = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            z2 = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            y_true = Variable(labels).to(device)

            # G1 results: 
            # visual_sample：generated visual feature from text feature
            visual_sample, real_eeg = netG(z, eeg_feat)
            # real loss
            D2_real, C2_real = netD2(real_eeg)
            D2_loss_real = torch.mean(D2_real)
            
            if is_aug:
                C2_loss_real = F.cross_entropy(C2_real[:batch_size], y_true) + lam * F.cross_entropy(C2_real[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_real[batch_size:], labels_b)
            else:
                C2_loss_real = F.cross_entropy(C2_real, y_true)
                
            DC2_loss = -D2_loss_real + clalambda * C2_loss_real 
            DC2_loss.backward(retain_graph=True)

            # fake loss
            # reduced text feature from visual_feature
            real_visual = netG2(z2, visual_sample).detach()
            D2_fake, C2_fake  = netD2(real_visual)
            D2_loss_fake = torch.mean(D2_fake)
            
            if is_aug:
                C2_loss_fake = F.cross_entropy(C2_fake[:batch_size], y_true) + lam * F.cross_entropy(C2_fake[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_fake[batch_size:], labels_b)
            else:
                C2_loss_fake = F.cross_entropy(C2_fake, y_true)
                
            DC2_loss = D2_loss_fake + clalambda * C2_loss_fake
            DC2_loss.backward(retain_graph=True)

            # train with gradient penalty (WGAN_GP)
            grad_penalty2 = calc_gradient_penalty(netD2, real_eeg.data, real_visual.data)
            grad_penalty2.backward(retain_graph=True)
            Wasserstein_D2 = D2_loss_real - D2_loss_fake
            optimizersCAN['optimizerD2'].step()
            reset_grad(nets)

        """Generator2"""
        for _ in range(1):
            eeg_feat = Variable(eeg_features)
            X = Variable(image_features)
            z = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            z2 = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            y_true = Variable(labels).to(device)
            

            _, eeg_feat = netG(z, eeg_feat)
            eeg_sample = netG2(z2, X)
            D2_fake, C2_fake = netD2(eeg_sample)
            _, C2_real = netD2(eeg_feat)
            
            # GAN's G loss
            G2_loss = torch.mean(D2_fake)
            
            if is_aug:
                C2_loss = (F.cross_entropy(C2_real[:batch_size], y_true) + F.cross_entropy(C2_fake[:batch_size], y_true))/2 + (lam * F.cross_entropy(C2_real[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_real[batch_size:], labels_b) + lam * F.cross_entropy(C2_fake[batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_fake[batch_size:], labels_b)) / 2
            else:
                C2_loss = (F.cross_entropy(C2_real, y_true) + F.cross_entropy(C2_fake, y_true))/2
            GC2_loss = -G2_loss + clalambda * C2_loss

            # ||W||_2 regularization (required)
            reg_loss2 = Variable(torch.Tensor([0.0])).to(device)
            if REG_W_LAMBDA != 0:
                for name, p in netG2.named_parameters():
                    if 'weight' in name:
                        reg_loss2 += p.pow(2).sum()
                reg_loss2.mul_(REG_W_LAMBDA)

            all_loss = GC2_loss  + reg_loss2
            all_loss.backward(retain_graph=True)
            optimizersCAN['optimizerG2'].step()
            reset_grad(nets)

        """Cycle Loss"""
        for _ in range(1):
            eeg_feat = Variable(eeg_features)
            X = Variable(image_features)
            z = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)
            z2 = Variable(torch.randn(eeg_features.shape[0], param.z_dim)).to(device)

            G_sample, eeg_feat = netG(z, eeg_feat)
            back_eeg_sample = netG2(z2, G_sample)
            
            # second branch img->eeg->img cycle_loss2
            G2_sample = netG2(z2, X) # img->eeg
            back_img_sample, _ = netG(z, G2_sample) # eeg(G2_sample:generated eeg)->img
            
            cycle_loss1 = cyclelambda * torch.nn.MSELoss()(eeg_feat, back_eeg_sample)
            cycle_loss2 = cyclelambda * torch.nn.MSELoss()(X, back_img_sample)
            cycle_loss = cyclelambda * ((cycle_loss1 + cycle_loss2)/ 2)
            
            label = torch.arange(eeg_features.shape[0]).to(device)
            logits = (eeg_features @ image_features.T) * torch.exp(torch.tensor(temperature))
            loss_i = F.cross_entropy(logits, label, reduction='none')
            loss_t = F.cross_entropy(logits.T, label, reduction='none')

            loss_cos = (loss_i + loss_t) / 2.0
            loss_cos = loss_cos.mean() 
            # total loss
            loss = cycle_loss + loss_cos
            print('Cycle_Loss',cycle_loss)
            print('CLIP_Loss',loss_cos)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            cycle_loss.backward(retain_graph=True)
            optimizersCAN['optimizerG'].step()
            optimizersCAN['optimizerG2'].step()
            reset_grad(nets)
    return loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train(model, temperature, optimizer, scheduler, START_EPOCH, num_epochs, experiment_num):
    for epoch in range(START_EPOCH, num_epochs):
        running_loss = 0.0
        EEG_embedding = np.array([])
        image_embeddings = np.array([])
        EEG_embedding_proj = np.array([])
        image_embeddings_proj = np.array([])
        labels_array = np.array([])
        logsoftmax   = nn.LogSoftmax(dim=-1)

        tq = tqdm(data_loader)
        for batch_idx, (EEGs, images, labels) in enumerate(tq):
            EEGs, images, labels = EEGs.to(device), images.to(device), labels.to(device)

            ######### Data augmentation(image mixup and EEG crop randomly but retain half) ########
            if is_aug:
                aug_image_feat, aug_eeg_feat, labels_a, labels_b, lam = augment_fuc(images, EEGs, labels,MixRatio)
                images = torch.cat((images,aug_image_feat), dim=0)
                EEGs = torch.cat((EEGs,aug_eeg_feat), dim=0)

            # images = preprocess(images)
            # get the embeddings for the EEG and images
            # optimizer.zero_grad()
            # bt,256
            EEG_embed, image_embed = model(EEGs, images) 
            # normalize the features
            EEG_embed = EEG_embed / EEG_embed.norm(dim=1, keepdim=True)
            image_embed = image_embed / image_embed.norm(dim=1, keepdim=True)
            
            if is_aug:
                loss = Cycle_GAN(image_embed,EEG_embed, labels, labels_a, labels_b, lam, temperature)
            else:
                loss = Cycle_GAN(image_embed,EEG_embed, labels, 0, 0, 0, temperature)

            # # backpropagate and update parameters
            # loss.backward()
            # optimizer.step()

            running_loss += loss.item()

            tq.set_description('[%d, %5d] total_loss: %.3f' %
                    (epoch + 1, batch_idx + 1, running_loss / (batch_idx+1.0)))
                
        if epoch == 32 or epoch == 64 or epoch == 128 or epoch == 200 or epoch == 256 or epoch == 300 or epoch == 400 or epoch == 512 or epoch == 600 or epoch == 700 or epoch == 800 or epoch == 900 or epoch == 1024 or epoch == 1100 or epoch == 1200 or epoch == 1300 or epoch == 1400 or epoch == 1500 or epoch == 1536 or epoch == 1600 or epoch == 1700 or epoch == 1800 or epoch == 1900 or epoch == 2000 or epoch == 2048 or epoch == 2200 or epoch == 2400 or epoch == 2560 or epoch == 2700 or epoch == 2900 or epoch == 3072 or epoch == 3584 or epoch == 4096:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, 'EEGClip_ckpt/EXPERIMENT_{}/checkpoints/clip_{}.pth'.format(experiment_num, epoch)) 
       
#load the data
## Training data
x_train_eeg = []
x_train_image = []
labels = []

for i in tqdm(natsorted(os.listdir(base_path + train_path))):
    loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1]
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    x_train_eeg.append(eeg_temp)
    img = loaded_array[0]
    # img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_train_image.append(img)
    labels.append(loaded_array[2])
    
x_train_eeg = np.array(x_train_eeg)
x_train_image = np.array(x_train_image)
x_train_image = np.squeeze(x_train_image)
labels = np.array(labels)

## Validation data
x_val_eeg = []
x_val_image = []
label_Val = []

for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
    loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1]
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    x_val_eeg.append(eeg_temp)
    img = loaded_array[0]
    # img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_val_image.append(img)
    label_Val.append(loaded_array[2])
    
x_val_eeg = np.array(x_val_eeg)
x_val_image = np.array(x_val_image)
x_val_image = np.squeeze(x_val_image)
labels_val = np.array(label_Val)

# ## hyperparameters
input_channels = args.input_channels
embedding_dim  = args.embedding_dim
projection_dim = args.projection_dim
input_size     = args.input_channels
hidden_size    = args.hidden_size
num_layers     = args.num_layers
batch_size     = args.batch_size
epoch          = args.epoch

# ## convert numpy array to tensor
x_train_eeg = torch.from_numpy(x_train_eeg).float().to(device)
x_train_image = torch.from_numpy(x_train_image).float().to(device)
labels = torch.from_numpy(labels).long().to(device)

# train_data  = CustomDataset(x_train_eeg, x_train_image, labels)

x_val_eeg = torch.from_numpy(x_val_eeg).float().to(device)
x_val_image = torch.from_numpy(x_val_image).float().to(device)
labels_val = torch.from_numpy(labels_val).long().to(device)
print(x_train_eeg.shape, x_train_image.shape, labels.shape, x_val_eeg.shape, x_val_image.shape, labels_val.shape)

train_data = torch.utils.data.TensorDataset(x_train_eeg, x_train_image, labels)
data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True, drop_last=True)
val_data = torch.utils.data.TensorDataset(x_val_eeg, x_val_image, labels_val)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=True, drop_last=True)

# ## define the model
# eeg_embedding = EEG_Encoder(projection_dim=projection_dim, num_layers=num_layers).to(device)
eeg_embedding = Enc_eeg().to(device)

# image_embedding = torchvision.models.resnet50(pretrained=True).to(device)
# weights = ResNet50_Weights.DEFAULT
# preprocess = weights.transforms().to(device)

# for param in image_embedding.parameters():
#     param.requires_grad = False

# num_features = image_embedding.fc.in_features

image_embedding = Proj_img().to(device)

model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

##################################################################################################
############################## CAN model for Cycle-consistency-loss###############################
##################################################################################################
lr=5*1e-5
is_aug=False
MixRatio=0.25
REG_W_LAMBDA=1e-3
cyclelambda=1
REG_Wz_LAMBDA=1e-4
GP_LAMBDA=10
CENT_LAMBDA=1
clalambda=1

param = _param()
netG = _netG(param.eeg_dim, param.X_dim).to(device)
netG.apply(weights_init)
netD = _netD(param.y_dim,param.X_dim).to(device)
netD.apply(weights_init)

netG2 = _netG2(param.X_dim).to(device)
netG2.apply(weights_init)
netD2 = _netD2(param.y_dim).to(device)
netD2.apply(weights_init)

# Optimizers for CANZSL
optimizersCAN = {
"optimizerD": optim.RMSprop(netD.parameters(), lr=lr, alpha=0.9),
"optimizerG": optim.RMSprop(netG.parameters(), lr=lr, alpha=0.9),
"optimizerD2": optim.RMSprop(netD2.parameters(), lr=lr, alpha=0.9),
"optimizerG2": optim.RMSprop(netG2.parameters(), lr=lr, alpha=0.9)
}
nets = [netG, netD, netG2, netD2] 
print('initial define done.')

scheduler = None
dir_info  = natsorted(glob('/data/lihao/workspace/MB2C/model/EEGCVPR40/EEGClip_ckpt/EXPERIMENT_*'))
if len(dir_info)==0:
    experiment_num = 1
else:
    experiment_num = int(dir_info[-1].split('_')[-1])  + 1

if not os.path.isdir(':/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}'.format(experiment_num)):
    os.makedirs(':/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}'.format(experiment_num))
    os.makedirs(':/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}/umap'.format(experiment_num))
    os.makedirs(':/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}/tsne'.format(experiment_num))
    os.system('cp *.py :/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}'.format(experiment_num))

ckpt_lst = natsorted(glob(':/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}/checkpoints/clip_*.pth'.format(experiment_num)))
print(experiment_num)
START_EPOCH = 0

if len(ckpt_lst)>=1:
    ckpt_path  = ckpt_lst[-1]
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    START_EPOCH = checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    START_EPOCH += 1
else:
    os.makedirs(':/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_{}/checkpoints/'.format(experiment_num))

train(model, 0.5, optimizer, scheduler, START_EPOCH, epoch, experiment_num)
print('completed')