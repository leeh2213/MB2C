import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class EEGDataset(Dataset):
    # Constructor
    def __init__(self, eeg_signals_path, subject = 1):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = '../datasets/DNN_feature_maps/full_feature_maps/clip/pretrained-True'
        
        self.num_voxels = 440
        self.data_len = 512
        # Compute size
        self.size = len(self.data)
        
    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        # eeg timesteps : 440
        eeg = eeg[20:460,:]
        ##### add preprocess and transpose
        eeg = np.array(eeg.transpose(0,1))
        eeg = torch.from_numpy(eeg).float()
        
        ##### preprocess
        label = torch.tensor(self.data[i]["label"]).long()
        # Get label---paried eeg and image
        image_name = self.images[self.data[i]["image"]]
        # ..\dreamdiffusion\datasets\DNN_feature_maps\full_feature_maps\clip\pretrained-True
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.npy')
        img_feat = np.load(image_path)
        return {'eeg': eeg, 'label': label, 'img_feat':  img_feat}
    

class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=1):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        # train_num:669 val_num:167 test_num:164 
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 440
        # self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


class CVPR40(Dataset):
    def __init__(self, args):
        
        self.args = args
        self.eeg_signals_path = args.eeg_signals_path
        self.splits_path = args.splits_path
        self.subject = args.subject
    
    def create_EEG_dataset(self):
        if self.subject == 0:
            self.splits_path = '../datasets/block_splits_by_image_all.pth'
        dataset      = EEGDataset(self.eeg_signals_path, self.subject)
        dataset_val  = EEGDataset(self.eeg_signals_path, self.subject)
        dataset_test = EEGDataset(self.eeg_signals_path, self.subject)
            
        eeg_latents_dataset_train = Splitter(dataset, split_path = self.splits_path, split_num = 0, split_name = 'train', subject= self.subject)
        eeg_latents_dataset_val   = Splitter(dataset_val, split_path = self.splits_path, split_num = 0, split_name = 'val', subject= self.subject)
        eeg_latents_dataset_test  = Splitter(dataset_test, split_path = self.splits_path, split_num = 0, split_name = 'test', subject = self.subject)
        
        dataloader = DataLoader(eeg_latents_dataset_train, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(eeg_latents_dataset_val, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(eeg_latents_dataset_test, batch_size=self.args.batch_size, shuffle=False)
        
        return (dataloader, val_loader, test_loader)
