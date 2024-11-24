import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import resample

def augment_fuc(args, image_data, eeg_data, labels):
    batch_size = image_data.shape[0]
    index = np.random.permutation(batch_size)
    mix_image = []
    mix_eeg = []
    labels_a, labels_b = [], []
    '''Returns mixed inputs, pairs of targets, and lambda'''
    alpha = 1.0
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    for i in range(int(batch_size * args.MixRatio)):
        if labels[i]!=labels[index[i]]:
            if len(mix_image)==0:
                mix_image = (lam * image_data[i] + (1 - lam) * image_data[index[i]]).unsqueeze(0)
                mix_eeg = (lam * eeg_data[i] + (1 - lam) * eeg_data[index[i]]).unsqueeze(0)
                labels_a = labels[i].unsqueeze(0)
                labels_b = labels[index[i]].unsqueeze(0)
            else:
                # image mixup
                mix_image = torch.cat((mix_image,(lam * image_data[i] + (1 - lam) * image_data[index[i]]).unsqueeze(0)), dim=0)
                mix_eeg = torch.cat((mix_eeg,(lam * eeg_data[i] + (1 - lam) * eeg_data[index[i]]).unsqueeze(0)), dim=0)
                labels_a = torch.cat((labels_a,labels[i].unsqueeze(0)), dim=0)
                labels_b = torch.cat((labels_b,labels[index[i]].unsqueeze(0)), dim=0)
        else:
            batch_size = batch_size + int(1/args.MixRatio)

        
        # down_sample= int(eeg_data.shape[-1]/2)
        # 1
        # down_sample1 = resample(eeg_data[i,:].detach().cpu().numpy(), down_sample).reshape(1,-1)
        # down_sample2 = resample(eeg_data[index[i],:].detach().cpu().numpy(), down_sample).reshape(1,-1)
        # down_sample1 = torch.from_numpy(down_sample1).cuda()
        # down_sample2 = torch.from_numpy(down_sample2).cuda()
        # 2
        # down_sample1 = ((eeg_data[i,:][::2] + eeg_data[i,:][1::2]) / 2).reshape(1,-1)
        # down_sample2 = ((eeg_data[index[i],:][::2] + eeg_data[index[i],:][1::2]) / 2).reshape(1,-1)

        # 3
    #     eeg_data = eeg_data.squeeze(1)
    #     down_sample1 = F.interpolate(eeg_data[i,:,:].unsqueeze(0), size=down_sample, mode='nearest').squeeze(dim=0)
    #     down_sample2 = F.interpolate(eeg_data[index[i],:,:].unsqueeze(0), size=down_sample, mode='nearest').squeeze(dim=0)  
    #     # eeg random crop and concat
    #     eeg_data[i,:] = torch.cat((down_sample1, down_sample2), dim=1)
    # eeg_data = eeg_data.unsqueeze(1)
    return mix_image, mix_eeg, labels_a, labels_b, lam
        

def crop_eeg(eeg_data):
    """
    Randomly crops the input EEG signal.
    Args:
        eeg: Input EEG signal of shape (n_channels, n_samples)
        crop_size: Tuple containing the size of the crop in (channels, samples)
    Returns:
        Cropped EEG signal of shape (n_channels, crop_size[1])
    """
    _, n_samples = eeg_data.shape
    crop_samples = eeg_data.size(1) // 2
    
    # Generate random indices for cropping
    crop_sample_idx = np.random.randint(0, n_samples - crop_samples + 1)

    # Crop the input signal
    cropped_eeg = eeg_data[:, crop_sample_idx:crop_sample_idx + crop_samples]

    # pad_size    = n_samples - cropped_eeg.shape[1]
    # cropped_eeg = np.pad(cropped_eeg, ((0, 0), (pad_size, 0)), mode="constant")

    return cropped_eeg
    