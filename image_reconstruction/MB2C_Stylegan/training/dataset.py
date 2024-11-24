# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import cv2
import config
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from torch.utils.data import Dataset
from network import *
import torchvision.transforms as transforms

try:
    import pyspng
except ImportError:
    pyspng = None


class EEG2ImageDataset(Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        print(super_kwargs)
        self.dataset_path = path
        self.eegs   = []
        self.images = []
        self.labels = []
        self.class_name = []
        self.eeg_feats= []
        self.img_feats = []
        self._raw_shape = [3, config.image_height, config.image_width] # 3,128,128
        self.resolution = config.image_height # 128
        self.has_labels  = True
        self.label_shape = [config.projection_dim] # 128
        self.label_dim   = config.projection_dim # 128
        self.name        = config.dataset_name 
        self.image_shape = [3, config.image_height, config.image_width] # 3,128,128
        self.num_channels = config.input_channel # 3
        is_cnn = config.is_cnn

        ## Loading Pre-trained Encoder #####
        checkpoint = torch.load('/data/lihao/workspace/MB2C/image_reconstruction/MB2C_Stylegan/eegbestckpt/clip_1600.pth', map_location=config.device)
        eeg_embedding = Enc_eeg().to(config.device)
        img_embedding = Proj_img().to(config.device)
        self.model = CLIPModel(eeg_embedding, img_embedding,).to(config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(config.device)
        self.img_encoder = self.model.image_encoder
        # self.eeg_encoder = self.model.eeg_encoder
        ########################################

        print('loading dataset...')
        for path in tqdm(natsorted(glob('/data/lihao/workspace/MB2C/image_reconstruction/MB2C_Stylegan/dataset/eeg_imagenet40_cvpr_2017_raw/train/*'))):
            loaded_array = np.load(path, allow_pickle=True)
            img = np.float32(cv2.resize(loaded_array[0], (config.image_height, config.image_width))) # 128,128,3
            self.images.append(np.transpose(img, (2, 0, 1)))
        
        for path in tqdm(natsorted(glob(self.dataset_path))):
            loaded_array = np.load(path, allow_pickle=True)

            if is_cnn == False:
                # eeg = loaded_array[1].T
                img = torch.from_numpy(loaded_array[0]).to(config.device)
            else:
                eeg = np.expand_dims(loaded_array[1].T, axis=0)
            # self.eegs.append(eeg)
            self.labels.append(loaded_array[2])
            # self.class_name.append(loaded_array[3])
            with torch.no_grad():
                # norm = np.max(eeg) / 2.0
                # eeg  = (eeg - norm) / norm # 440,128
                # eeg_feat = self.eeg_encoder(eeg)
                img_feat = self.img_encoder(img)
                self.img_feats.append(img_feat.detach().cpu().numpy())
                # self.eeg_feats.append(self.eeg_encoder(eeg_feat).to(config.device)).detach().cpu().numpy()


        # self.eegs     = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
        self.images   = torch.from_numpy(np.array(self.images)).to(torch.float32) # [7947, 3, 128, 128]
        self.img_feats = torch.from_numpy(np.array(self.img_feats)).squeeze(1).to(torch.float32) # [7947, 768]
        # self.eeg_feat = torch.from_numpy(np.array(self.eeg_feat)).to(torch.float32)
        self.labels   = torch.from_numpy(np.array(self.labels)).to(torch.int32) # 7947


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # eeg   = self.eegs[idx]
        # norm  = torch.max(eeg) / 2.0
        # eeg   =  ( eeg - norm ) / norm
        # image = self.images[idx]
        image = self.images[idx]
        label = self.labels[idx]
        con   = self.img_feats[idx]
        return image, con
    
    def get_label(self, idx):
        # con = self.eeg_feat[idx]
        con = self.img_feats[idx]
        return con

class Image2EEG2ImageDataset(Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        print(super_kwargs)
        self.dataset_path = path
        self.eegs   = []
        self.images = []
        self.labels = []
        self.class_name = []
        self.eeg_feat = np.array([])
        temp_images   = []
        self.subject_num = []
        cls_lst = [0, 1]
        self._raw_shape = [3, config.image_height, config.image_width]
        self.resolution = config.image_height
        self.has_labels  = True
        self.label_shape = [config.projection_dim]
        self.label_dim   = config.projection_dim
        self.name        = config.dataset_name
        self.image_shape = [3, config.image_height, config.image_width]
        self.num_channels = config.input_channel
        # self.preprocess   = GoogLeNet_Weights.IMAGENET1K_V1.transforms()
        self.preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])

        # ## Loading Pre-trained EEG Encoder #####
        # self.eeg_model = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim, num_layers=config.num_layers).to(config.device)
        # self.eeg_model = torch.nn.DataParallel(self.eeg_model).to(config.device)
        # eegckpt   = 'eegbestckpt/eegfeat_all_0.9665178571428571.pth'
        # eegcheckpoint = torch.load(eegckpt, map_location=config.device)
        # self.eeg_model.load_state_dict(eegcheckpoint['model_state_dict'])
        # print('Loading EEG checkpoint from previous epoch: {}'.format(eegcheckpoint['epoch']))
        # ########################################

        ## Loading Pre-trained Image Encoder #####
        self.image_model     = ImageFeatNet(projection_dim=config.projection_dim).to(config.device)
        self.image_model     = torch.nn.DataParallel(self.image_model).to(config.device)
        ckpt_path = 'imageckpt/eegfeat_all_0.6875.pth'
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        self.image_model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading Image checkpoint from previous epoch: {}'.format(checkpoint['epoch']))
        ########################################

        print('loading dataset...')
        for path in tqdm(natsorted(glob(self.dataset_path))):
            loaded_array = np.load(path, allow_pickle=True)
            # if loaded_array[2] in cls:
            self.eegs.append(loaded_array[1].T)
            # self.eegs.append(np.expand_dims(loaded_array[1].T, axis=0))
            self.images.append(np.transpose(np.float32(cv2.resize(loaded_array[0], (config.image_height, config.image_width))), (2, 0, 1)))
            self.labels.append(loaded_array[2])
            self.class_name.append(loaded_array[3])
            self.subject_num.append(loaded_array[4]) # Subject Number

            img = np.float32(loaded_array[0])
            img = self.preprocess( img ).numpy()
            c   = np.zeros(shape=(config.n_subjects,), dtype=np.float32)
            c[loaded_array[4]-1] = 1.0
            c 	= np.expand_dims( np.expand_dims(c, axis=-1), axis=-1 )
            c 	= np.tile(c, (1, img.shape[1], img.shape[2]))
            img = np.concatenate([img, c], axis=0)
            temp_images.append(img)
        
        temp_images = torch.from_numpy(np.array(temp_images)).to(torch.float32)
        for idx in tqdm(range(0, temp_images.shape[0], 256)):
            batch_images = temp_images[idx:idx+256].to(config.device)
            with torch.no_grad():
                feat  = (self.image_model(batch_images)).detach().cpu().numpy()
            self.eeg_feat = np.concatenate((self.eeg_feat, feat), axis=0) if self.eeg_feat.size else feat
        
        print(self.eeg_feat.shape)
        
        k_means             = K_means(n_clusters=config.n_classes)
        clustering_acc_proj = k_means.transform(np.array(self.eeg_feat), np.array(self.labels))
        print("[Test KMeans score Proj: {}]".format(clustering_acc_proj))

        self.eegs        = torch.from_numpy(np.array(self.eegs)).to(torch.float32)
        self.images      = torch.from_numpy(np.array(self.images)).to(torch.float32)
        self.eeg_feat    = torch.from_numpy(np.array(self.eeg_feat)).to(torch.float32)
        self.labels      = torch.from_numpy(np.array(self.labels)).to(torch.int32)
        self.subject_num = torch.from_numpy(np.array(self.subject_num)).to(torch.int32)
        # self.class_name = torch.from_numpy(np.array(self.class_name))


    def __len__(self):
        return self.eegs.shape[0]

    def __getitem__(self, idx):
        eeg   = self.eegs[idx]
        norm  = torch.max(eeg) / 2.0
        eeg   =  ( eeg - norm ) / norm
        image = self.images[idx]
        label = self.labels[idx]
        con   = self.eeg_feat[idx]
        # class_n = self.class_name[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        # return eeg, image, label, con, class_n
        return image, con
    
    def get_label(self, idx):
        # label = self._get_raw_labels()[self._raw_idx[idx]]
        # if label.dtype == np.int64:
        #     onehot = np.zeros(self.label_shape, dtype=np.float32)
        #     onehot[label] = 1
        #     label = onehot
        # label = self.labels[idx]
        # con   = np.zeros(shape=(40,), dtype=np.float32)
        # con[label.numpy()] = 1.0
        # con   = torch.from_numpy(con)
        con = self.eeg_feat[idx]
        return con