import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable


from tqdm import tqdm
from augmentation.augment import augment_fuc
from end_to_end_encoding_utils import *
from Loss.ganModelCls import _param,_netG,_netD,_netG2,_netD2
from EEG_Encoder.SelfModel import *
from encoder_modules import *
from torch.utils.tensorboard import SummaryWriter

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = 'results/' 
model_idx = 'CVPR40_encoder_training_from_scratch_clip'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='1000', type=int)
parser.add_argument('-batch_size', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training.')

parser.add_argument('--reproduce', type=bool, default=False)

# data augmentation
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--MixRatio', type=float, default=0.75)
parser.add_argument('--is_aug', type=bool, default=False)

# CANZSL 
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--disp_interval', type=int, default=10)
parser.add_argument('--is_gan', type=bool, default=False)
parser.add_argument('--cyclelambda', type=float, default=1000)
parser.add_argument('--REG_W_LAMBDA', type=float, default=1e-3)
parser.add_argument('--REG_Wz_LAMBDA', type=float, default=1e-4)
parser.add_argument('--GP_LAMBDA', type=float, default=10)   
parser.add_argument('--CENT_LAMBDA', type=float, default=1)  
parser.add_argument('--clalambda',  type=float, default=1)
parser.add_argument('--lr', type=float, default=3*1e-4)

# CVPR40 dataset settings
# E:\EEG Research\multi-modal\nice-eeg\Data\CVPR40
parser.add_argument('--project_dir', default='Data/CVPR40', type=str)
parser.add_argument('--eeg_signals_path', default= 'Data/CVPR40/eeg_5_95_std.pth', type=str)
parser.add_argument('--splits_path', default= 'Data/CVPR40/block_splits_by_image_single.pth', type=str)
parser.add_argument('--subject', default=0, type=int) 
# all if subject = 0
# self.splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth'
 
class IE():
    def __init__(self, args):
        super(IE, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 3*1e-4
        self.b1 = 0.5
        self.b2 = 0.9999

        self.start_epoch = 0
        self.eeg_data_path = 'Data/Things-EEG2/Preprocessed_data_250Hz/'
        self.img_data_path = 'Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/' + args.dnn + '/pretrained-True/'
        self.test_center_path = 'Data/Things-EEG2/Image_set/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject.txt", 'a+')
        self.writer = SummaryWriter(log_dir=result_path+'/log/')
        self.early_stopping = EarlyStopping(patience=1000, verbose=True)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().to(args.device)
        self.Enc_eeg = Enc_eeg().to(args.device)
        self.Proj_eeg = Proj_eeg().to(args.device)
        self.Proj_img = Proj_img().to(args.device)
        
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])
        print('args.pretrained:',args.pretrained)
        if args.pretrained:
            self.Enc_eeg.load_state_dict(torch.load('model/' + model_idx + 'Enc_eeg_cls.pth'))
            self.Proj_eeg.load_state_dict(torch.load('model/' + model_idx + 'Proj_eeg_cls.pth'))
            self.Proj_img.load_state_dict(torch.load('model/' + model_idx  + 'Proj_img_cls.pth'))
        ##################################################################################################
        ############################## CAN model for Cycle-consistency-loss###############################
        ##################################################################################################
        self.param = _param()
        self.netG = _netG(self.param.eeg_dim, self.param.X_dim).to(args.device)
        self.netG.apply(self.weights_init)
        print('netG:  ',self.netG)
        self.netD = _netD(self.param.y_dim,self.param.X_dim).to(args.device)
        self.netD.apply(self.weights_init)
        print('netD:  ',self.netD)

        self.netG2 = _netG2(self.param.X_dim).to(args.device)
        self.netG2.apply(self.weights_init)
        print('netG2:  ',self.netG2)
        self.netD2 = _netD2(self.param.y_dim).to(args.device)
        self.netD2.apply(self.weights_init)
        print('netD2:  ',self.netD2)
        
        # Optimizers for CANZSL
        self.optimizersCAN = {
        "optimizerD": optim.RMSprop(self.netD.parameters(), lr=self.args.lr, alpha=0.9),
        "optimizerG": optim.RMSprop(self.netG.parameters(), lr=self.args.lr, alpha=0.9),
        "optimizerD2": optim.RMSprop(self.netD2.parameters(), lr=self.args.lr, alpha=0.9),
        "optimizerG2": optim.RMSprop(self.netG2.parameters(), lr=self.args.lr, alpha=0.9)
        }
        self.nets = [self.netG, self.netD, self.netG2, self.netD2] 
        ##################################################################################################
        # Optimizers for clip loss
        # if not args.pretrained:
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def TrainOneEpoch(self, args, epoch, image_features, eeg_features, labels, labels_a, labels_b, lam):
        
        with torch.autograd.set_detect_anomaly(True):
            """"
            Clip Loss
            cosine similarity as the logits
            """
            # cosine similarity as the logits
            fake_labels = torch.arange(eeg_features.shape[0])  # used for the loss
            fake_labels = Variable(fake_labels.cuda().type(self.LongTensor))
            logit_scale = self.logit_scale.exp()
            logits_per_eeg = logit_scale * eeg_features @ image_features.t()
            logits_per_img = logits_per_eeg.t()

            loss_eeg = self.criterion_cls(logits_per_eeg, fake_labels)
            loss_img = self.criterion_cls(logits_per_img, fake_labels)
            loss_cos = (loss_eeg + loss_img) / 2
            
            if args.is_gan:
                """ Discriminator """
                for _ in range(5):
                    eeg_feat = eeg_features
                    X = Variable(image_features)
                    z = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    y_true = Variable(labels).to(self.args.device)

                    D_real, C_real = self.netD(X) 
                    D_loss_real = torch.mean(D_real) 
                    if args.is_aug:
                        C_loss_real = F.cross_entropy(C_real[:self.args.batch_size], y_true) + lam * F.cross_entropy(C_real[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_real[self.args.batch_size:], labels_b)
                    else:
                        C_loss_real = F.cross_entropy(C_real, y_true)
                        
                    DC_loss = -D_loss_real + C_loss_real
                    DC_loss.backward(retain_graph=True)
                    
                    G_sample, _ = self.netG(z, eeg_feat)
                    
                    D_fake, C_fake = self.netD(G_sample)
                    D_loss_fake = torch.mean(D_fake)
                    
                    if args.is_aug:
                        C_loss_fake = F.cross_entropy(C_fake[:self.args.batch_size], y_true) + lam * F.cross_entropy(C_fake[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_fake[self.args.batch_size:], labels_b)
                    else:
                        C_loss_fake = F.cross_entropy(C_fake, y_true)
                        
                    DC_loss = D_loss_fake + C_loss_fake
                    DC_loss.backward(retain_graph=True)

                    # train with gradient penalty (WGAN_GP)
                    grad_penalty = self.calc_gradient_penalty(self.netD, X.data, G_sample.data)
                    grad_penalty.backward(retain_graph=True)

                    # Wasserstein Loss
                    Wasserstein_D = D_loss_real - D_loss_fake
                    self.optimizersCAN['optimizerD'].step()
                    self.reset_grad(self.nets)

                """ Generator """
                for _ in range(1):
                    eeg_feat = Variable(eeg_features)
                    X = Variable(image_features)
                    z = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    y_true = Variable(labels).to(self.args.device)

                    G_sample, _ = self.netG(z, eeg_feat)
                    D_fake, C_fake = self.netD(G_sample)
                    _, C_real = self.netD(X)
                    # GAN's G loss
                    G_loss = torch.mean(D_fake)
                    if args.is_aug:
                        C_loss = (F.cross_entropy(C_real[:self.args.batch_size], y_true) + F.cross_entropy(C_fake[:self.args.batch_size], y_true)) / 2 + (lam * F.cross_entropy(C_real[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_real[self.args.batch_size:], labels_b) + lam * F.cross_entropy(C_fake[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C_fake[self.args.batch_size:], labels_b)) / 2
                    else:
                        C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2
                    
                    GC_loss = -G_loss + C_loss
                    
                    reg_loss = Variable(torch.Tensor([0.0])).to(self.args.device)
                    if self.args.REG_W_LAMBDA != 0:
                        for name, p in self.netG.named_parameters():
                            if 'weight' in name:
                                reg_loss += p.pow(2).sum()
                        reg_loss.mul_(self.args.REG_W_LAMBDA)
                        
                    all_loss = GC_loss + reg_loss
                    all_loss.backward(retain_graph=True)
                    self.optimizersCAN['optimizerG'].step()
                    self.reset_grad(self.nets)
                    
                """Discriminator2"""
                for _ in range(5):
                    eeg_feat = Variable(eeg_features)
                    X = Variable(image_features)
                    z = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    z2 = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    y_true = Variable(labels).to(self.args.device)

                    # G1 results: 
                    # visual_sample：generated visual feature from text feature
                    visual_sample, real_eeg = self.netG(z, eeg_feat)
                    # real loss
                    D2_real, C2_real = self.netD2(real_eeg)
                    D2_loss_real = torch.mean(D2_real)
                    
                    if args.is_aug:
                        C2_loss_real = F.cross_entropy(C2_real[:self.args.batch_size], y_true) + lam * F.cross_entropy(C2_real[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_real[self.args.batch_size:], labels_b)
                    else:
                        C2_loss_real = F.cross_entropy(C2_real, y_true)
                        
                    DC2_loss = -D2_loss_real + self.args.clalambda * C2_loss_real 
                    DC2_loss.backward(retain_graph=True)

                    # fake loss
                    # reduced text feature from visual_feature
                    real_visual = self.netG2(z2, visual_sample).detach()
                    D2_fake, C2_fake  = self.netD2(real_visual)
                    D2_loss_fake = torch.mean(D2_fake)
                    
                    if args.is_aug:
                        C2_loss_fake = F.cross_entropy(C2_fake[:self.args.batch_size], y_true) + lam * F.cross_entropy(C2_fake[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_fake[self.args.batch_size:], labels_b)
                    else:
                        C2_loss_fake = F.cross_entropy(C2_fake, y_true)
                        
                    DC2_loss = D2_loss_fake + self.args.clalambda * C2_loss_fake
                    DC2_loss.backward(retain_graph=True)

                    # train with gradient penalty (WGAN_GP)
                    grad_penalty2 = self.calc_gradient_penalty(self.netD2, real_eeg.data, real_visual.data)
                    grad_penalty2.backward(retain_graph=True)
                    Wasserstein_D2 = D2_loss_real - D2_loss_fake
                    self.optimizersCAN['optimizerD2'].step()
                    self.reset_grad(self.nets)

                """Generator2"""
                for _ in range(1):
                    eeg_feat = Variable(eeg_features)
                    X = Variable(image_features)
                    z = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    z2 = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    y_true = Variable(labels).to(self.args.device)
                    

                    _, eeg_feat = self.netG(z, eeg_feat)
                    eeg_sample = self.netG2(z2, X)
                    D2_fake, C2_fake = self.netD2(eeg_sample)
                    _, C2_real = self.netD2(eeg_feat)
                    
                    # GAN's G loss
                    G2_loss = torch.mean(D2_fake)
                    
                    if args.is_aug:
                        C2_loss = (F.cross_entropy(C2_real[:self.args.batch_size], y_true) + F.cross_entropy(C2_fake[:self.args.batch_size], y_true))/2 + (lam * F.cross_entropy(C2_real[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_real[self.args.batch_size:], labels_b) + lam * F.cross_entropy(C2_fake[self.args.batch_size:], labels_a) + (1 - lam) * F.cross_entropy(C2_fake[self.args.batch_size:], labels_b)) / 2
                    else:
                        C2_loss = (F.cross_entropy(C2_real, y_true) + F.cross_entropy(C2_fake, y_true))/2
                    GC2_loss = -G2_loss + self.args.clalambda * C2_loss

                    # ||W||_2 regularization (required)
                    reg_loss2 = Variable(torch.Tensor([0.0])).to(self.args.device)
                    if self.args.REG_W_LAMBDA != 0:
                        for name, p in self.netG2.named_parameters():
                            if 'weight' in name:
                                reg_loss2 += p.pow(2).sum()
                        reg_loss2.mul_(self.args.REG_W_LAMBDA)

                    all_loss = GC2_loss  + reg_loss2
                    all_loss.backward(retain_graph=True)
                    self.optimizersCAN['optimizerG2'].step()
                    self.reset_grad(self.nets)

                """Cycle Loss"""
                for _ in range(1):
                    eeg_feat = Variable(eeg_features)
                    X = Variable(image_features)
                    z = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)
                    z2 = Variable(torch.randn(eeg_features.shape[0], self.param.z_dim)).to(self.args.device)

                    G_sample, eeg_feat = self.netG(z, eeg_feat)
                    back_eeg_sample = self.netG2(z2, G_sample)
                    
                    # second branch img->eeg->img cycle_loss2
                    G2_sample = self.netG2(z2, X) # img->eeg
                    back_img_sample, _ = self.netG(z, G2_sample) # eeg(G2_sample:generated eeg)->img
                    
                    cycle_loss1 = self.args.cyclelambda * torch.nn.MSELoss()(eeg_feat, back_eeg_sample)
                    cycle_loss2 = self.args.cyclelambda * torch.nn.MSELoss()(X, back_img_sample)
                    cycle_loss = self.args.cyclelambda * ((cycle_loss1 + cycle_loss2)/ 2)
                              
                    # total loss
                    loss = cycle_loss + loss_cos
                    
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    
                    cycle_loss.backward(retain_graph=True)
                    self.optimizersCAN['optimizerG'].step()
                    self.optimizersCAN['optimizerG2'].step()
                    self.reset_grad(self.nets)
                    
                    self.writer.add_scalars("Total Training Loss", {"total_val_loss":loss,"cycle_loss":cycle_loss,"infoNCEloss":loss_cos}, global_step=epoch)

                if epoch % self.args.disp_interval == 0 and epoch:
                    self.writer.add_scalars("Total Training Loss(GAN model)", {"Wasserstein_D":Wasserstein_D.item(),"reg_loss":reg_loss.item(),
                                                                              "G_loss":G_loss.item(),"D_loss_real":D_loss_real.item(),
                                                                              "D_loss_fake":D_loss_fake.item(),"Wasserstein_D2":Wasserstein_D2.item(),
                                                                              "reg_loss2":reg_loss2.item(),"G2_loss":G2_loss.item(),
                                                                              "D2_loss_real":D2_loss_real.item(),"D2_loss_fake":D2_loss_fake.item()}, global_step=epoch)
                    log_text = 'Iter-{}; Was_D: {:.4}; reg_ls: {:.4}; G_loss: {:.4}; ' \
                            'D_loss_real: {:.4}; D_loss_fake: {:.4}; \n' \
                            'Was_D2: {:.4}; reg_ls2: {:.4};  G2_loss: {:.4}; ' \
                            'D2_loss_real: {:.4}; D2_loss_fake: {:.4};Cyc_loss+InfoLoss: {:.4}; Cyc_loss: {:.4};\n'\
                        .format(epoch, Wasserstein_D.item(), reg_loss.item(),
                                G_loss.item(), D_loss_real.item(), D_loss_fake.item(),
                                Wasserstein_D2.item(), reg_loss2.item(), G2_loss.item(), D2_loss_real.item(),
                                D2_loss_fake.item(),loss.item(),cycle_loss.item())
                    print(log_text)      
      
    def ValOneEpoch(self, args, epoch, test_loader):
        # validation using test_eeg(test_center brain inference)
        if (epoch + 1) % 1 == 0:
            self.Enc_eeg.eval()
            self.Proj_eeg.eval()
            self.Proj_img.eval()
            if self.args.is_gan:
                self.netG.eval()
                self.netG2.eval()
            val_loss_ave = Averager()
            top1_Averager = Averager()
            top3_Averager = Averager()
            top5_Averager = Averager()
            with torch.no_grad():
                # validation part
                for i, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f'Epoch {epoch + 1}/{self.n_epochs}', leave=False):
                    veeg, _, vimg = batch['eeg'], batch['label'], batch['img_feat']
                    veeg = Variable(veeg.to(args.device).type(self.Tensor))
                    vimg_features = Variable(vimg.to(args.device).type(self.Tensor))
                    
                    vlabels = torch.arange(veeg.shape[0])
                    vlabels = Variable(vlabels.to(args.device).type(self.LongTensor))

                    veeg_features = self.Enc_eeg(veeg)
                    veeg_features = self.Proj_eeg(veeg_features)
                    vimg_features = self.Proj_img(vimg_features)

                    veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                    vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                    # Cycle-consistency-loss
                    if args.is_gan:
                        veeg_feat = Variable(veeg_features)
                        z = Variable(torch.randn(veeg_feat.shape[0], self.param.z_dim)).to(self.args.device)
                        z2 = Variable(torch.randn(veeg_feat.shape[0], self.param.z_dim)).to(self.args.device)

                        G_sample, _ = self.netG(z, veeg_feat)
                        back_eeg_sample = self.netG2(z2, G_sample)
                        
                        # second branch img->eeg->img cycle_loss2
                        G2_sample = self.netG2(z2, vimg_features) # img->eeg
                        back_img_sample, _ = self.netG(z, G2_sample) # eeg(G2_sample:generated eeg)->img
                
                        cycle_loss1 = self.args.cyclelambda * torch.nn.MSELoss()(veeg_feat, back_eeg_sample)
                        cycle_loss2 = self.args.cyclelambda * torch.nn.MSELoss()(vimg_features, back_img_sample)
                        cycle_loss = (cycle_loss1 + cycle_loss2)/ 2
                        
                        # cosine similarity as the logits
                        logit_scale = self.logit_scale.exp()
                        logits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        logits_per_img = logits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(logits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(logits_per_img, vlabels)
                        vloss_cos = (vloss_eeg + vloss_img) / 2
                        
                        total_val_loss = cycle_loss + vloss_cos
                        val_loss_ave.add(total_val_loss)
                        # 使用 add_scalar 方法记录损失，每个参数都在其自己的命名空间下
                        self.writer.add_scalars("Total Validation Loss", {"total_val_loss":val_loss_ave.item(),"cycle_loss":cycle_loss,"infoNCEloss":vloss_cos}, global_step=epoch)
                    else:
                        # cosine similarity as the logits
                        logit_scale = self.logit_scale.exp()
                        logits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        logits_per_img = logits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(logits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(logits_per_img, vlabels)
                        vloss_cos = (vloss_eeg + vloss_img) / 2
                        
                        total_val_loss = vloss_cos
                        val_loss_ave.add(total_val_loss)
                        # 使用 add_scalar 方法记录损失，每个参数都在其自己的命名空间下
                        # self.writer.add_scalars("Total Validation Loss:", {"total_val_loss":val_loss_ave.item(), "infoNCEloss":vloss_cos}, global_step=epoch)
                
                for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Epoch {epoch + 1}/{self.n_epochs}', leave=False):
                    test_eeg, test_labels , img_feat = batch['eeg'], batch['label'], batch['img_feat']
                    proj_test_center = self.Proj_img(img_feat)
                    norm_test_center = proj_test_center / proj_test_center.norm(dim=1, keepdim=True)
                    top1, top3, top5 = run_classification_test_labels(norm_test_center, self.Proj_eeg(self.Enc_eeg(Variable(test_eeg.type(self.Tensor)))), test_labels, (epoch+1))
                    top1_Averager.add(top1)
                    top3_Averager.add(top3)
                    top5_Averager.add(top5)
                    
                print("Test ACC Epoch:{} top1:{} top3:{} top5:{}".format(epoch, top1_Averager.item(), 
                                                                            top3_Averager.item(), 
                                                                            top5_Averager.item()))
                
                self.log_write.write('Epoch %d: , top1: %.4f ,top3: %.4f ,top5: %.4f\n'%((epoch + 1), top1_Averager.item() ,
                                                                                            top3_Averager.item(),
                                                                                            top5_Averager.item()))
                # 使用 SummaryWriter 记录测试准确率
                self.writer.add_scalars("Test ACC with best model", {"Top-1": top1_Averager.item(), "Top-3": top3_Averager.item(), "Top-5": top5_Averager.item()}, global_step=epoch)
                    
                    
                if val_loss_ave.item() <= self.best_loss_val:
                    self.best_loss_val = val_loss_ave.item()
                    self.best_epoch = epoch + 1
                    torch.save(self.Enc_eeg.state_dict(), 'model/' + model_idx  + 'Enc_eeg_cls.pth')
                    torch.save(self.Proj_eeg.state_dict(), 'model/' + model_idx  + 'Proj_eeg_cls.pth')
                    torch.save(self.Proj_img.state_dict(), 'model/' + model_idx  + 'Proj_img_cls.pth')
                    torch.save(self.netG.state_dict(), 'model/' + model_idx  + 'netG_cls.pth')
                    torch.save(self.netG2.state_dict(), 'model/' + model_idx  + 'netG2_cls.pth')
                    print('Best model saved!')
               
                 
            if args.is_gan:
                self.log_write.write('Epoch %d: , loss val: %.4f ,cyc_loss: %.4f\n'%((epoch + 1), total_val_loss.detach().cpu().numpy(), cycle_loss.detach().cpu().numpy()))
                print('Epoch:', (epoch + 1),
                    '  loss val: %.4f' % total_val_loss.detach().cpu().numpy(), '  cycle_loss val: %.4f' % cycle_loss.detach().cpu().numpy()
                    )
            else:
                self.log_write.write('Epoch %d: , loss val: %.4f \n'%((epoch + 1), total_val_loss.detach().cpu().numpy()))
                print('Epoch:', (epoch + 1),
                    '  loss val: %.4f' % total_val_loss.detach().cpu().numpy())
       
    def weights_init(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0.0)

    def reset_grad(self, nets):
        for net in nets:
            net.zero_grad()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.args.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.args.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.args.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.GP_LAMBDA
        return gradient_penalty
  
    def train(self, args, seed_n):
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        if not args.pretrained:
            self.Enc_eeg.apply(weights_init_normal)
            self.Proj_eeg.apply(weights_init_normal)
            self.Proj_img.apply(weights_init_normal)
            
        CVPR40Dataset = CVPR40(args)
        self.dataloader, self.val_loader, self.test_loader = CVPR40Dataset.create_EEG_dataset()
        self.best_loss_val = np.inf

        if not args.reproduce:
            for epoch in tqdm(range(self.n_epochs)):
                self.Enc_eeg.train()
                self.Proj_eeg.train()
                self.Proj_img.train()
                self.netG.train()
                self.netG2.train()

                for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f'Epoch {epoch + 1}/{self.n_epochs}', leave=False):
                    eeg, labels, img_feat = batch['eeg'], batch['label'], batch['img_feat']
                    eeg = Variable(eeg.to(args.device).type(self.Tensor))
                    labels = Variable(labels.to(args.device).type(self.LongTensor)) # used for the loss
                    img_features = Variable(img_feat.to(args.device).type(self.Tensor))
                    #######################################################################################
                    ######### Data augmentation(image mixup and EEG crop randomly but retain half) ########
                    if args.is_aug:
                        aug_image_feat, aug_eeg_feat, labels_a, labels_b, lam = augment_fuc(args, img_features, eeg, labels)
                        img_features = torch.cat((img_features,aug_image_feat), dim=0)
                        eeg = torch.cat((eeg,aug_eeg_feat), dim=0)
                    #######################################################################################
                    # obtain the features
                    eeg_features = self.Enc_eeg(eeg)

                    # project the features to a multimodal embedding space
                    eeg_features = self.Proj_eeg(eeg_features)
                    img_features = self.Proj_img(img_features)
                    
                    # normalize the features
                    eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                    img_features = img_features / img_features.norm(dim=1, keepdim=True)
                    
                    # cycle-consistency-loss + GAN loss + clip loss
                    if args.is_aug:
                        self.TrainOneEpoch(self.args, epoch, img_features, eeg_features, labels, labels_a, labels_b, lam)
                    else:
                        self.TrainOneEpoch(self.args, epoch, img_features, eeg_features, labels, labels, labels, 0.5)
                    
                # validation using test_eeg(test_center brain inference)
                self.ValOneEpoch(args, epoch, self.test_loader)    
                
                if self.early_stopping(self.best_loss_val):
                    print("Early stopping")
                    break
                    
        # * test part
        self.Enc_eeg.load_state_dict(torch.load('model/' + model_idx  + 'Enc_eeg_cls.pth'))
        self.Proj_eeg.load_state_dict(torch.load('model/' + model_idx  + 'Proj_eeg_cls.pth'))
        self.Proj_img.load_state_dict(torch.load('model/' + model_idx  + 'Proj_img_cls.pth'))

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            test_top1_ave =Averager()
            test_top3_ave =Averager()
            test_top5_ave =Averager()
            for _, batch in enumerate(self.test_loader):
                eeg, test_labels, img_feat = batch['eeg'], batch['label'], batch['img_feat']
                eeg_feat = self.Proj_eeg(self.Enc_eeg(Variable(eeg.type(self.Tensor))))
                proj_test_center = self.Proj_img(img_feat)
                norm_test_center = proj_test_center / proj_test_center.norm(dim=1, keepdim=True)
                top1_acc, top3_acc, top5_acc = run_classification_test_labels(norm_test_center, eeg_feat, test_labels ,0)
                test_top1_ave.add(top1_acc)
                test_top3_ave.add(top3_acc)
                test_top5_ave.add(top5_acc)

        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (test_top1_ave.item(), test_top3_ave.item(), test_top5_ave.item()))
        if args.reproduce:
            self.log_write.write('The best epoch is: %d\n' % self.best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (test_top1_ave.item(), test_top3_ave.item(), test_top5_ave.item()))
        self.writer.close()
        return test_top1_ave.item(), test_top3_ave.item(), test_top5_ave.item()

def main():
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
    starttime = datetime.now()
    # seed_n = np.random.randint(args.seed)
    seed_n = 42

    print('seed is ' + str(seed_n))
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    ie = IE(args)
    Acc, Acc3, Acc5 = ie.train(args, seed_n)
    print('THE BEST ACCURACY IS ' + str(Acc))


    endtime = datetime.datetime.now()
    print('duration: ' + str(endtime - starttime))

    aver.append(Acc)
    aver3.append(Acc3)
    aver5.append(Acc5)
    
    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))
    print('aver:' , aver)
    print('aver3:', aver3)
    print('aver5:', aver5)

    # column = np.arange(1, cal_num+1).tolist()
    # column.append('ave')
    # pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    # pd_all.to_csv(result_path + 'result.csv', mode='a',)
    
if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))