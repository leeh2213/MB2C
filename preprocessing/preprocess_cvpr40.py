from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from configs import cvpr40config as config


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

args = config.args
base_path       = args.base_path
train_path      = args.train_path
validation_path = args.validation_path
test_path       = args.test_path

for i in tqdm(natsorted(os.listdir(base_path + train_path))):
    loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
    filter_eeg = butter_bandpass_filter(loaded_array[1], 5, 95, 1000, order=2)
    loaded_array[1] = filter_eeg
    np.save(os.path.join('/data/lihao/workspace/MB2C/Data/CVPR40/my_5_95hz/clip/train/', os.path.basename(i)), loaded_array)

for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
    loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
    filter_eeg = butter_bandpass_filter(loaded_array[1], 5, 95, 1000, order=2)
    loaded_array[1] = filter_eeg
    np.save(os.path.join('/data/lihao/workspace/MB2C/Data/CVPR40/clip/val/', os.path.basename(i)), loaded_array)

for i in tqdm(natsorted(os.listdir(base_path + test_path))):
    loaded_array = np.load(base_path + test_path + i, allow_pickle=True)
    filter_eeg = butter_bandpass_filter(loaded_array[1], 5, 95, 1000, order=2)
    loaded_array[1] = filter_eeg
    np.save(os.path.join('/data/lihao/workspace/MB2C/Data/CVPR40/clip/test/', os.path.basename(i)), loaded_array)

print('complete all!')

