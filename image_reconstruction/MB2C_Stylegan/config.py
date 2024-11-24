import torch
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EEG Data
train_data_path = '/data/lihao/workspace/MB2C/image_reconstruction/MB2C_Stylegan/dataset/clip/train/*'
val_data_path   = '/data/lihao/workspace/MB2C/image_reconstruction/MB2C_Stylegan/dataset/clip/val/*'
test_data_path  = '/data/lihao/workspace/MB2C/image_reconstruction/MB2C_Stylegan/dataset/clip/test/*'

image_height = 128 #224
image_width  = 128 #224
input_channel= 3
kernel_size  = 3
padding      = 1
batch_size   = 32 #256#256
num_workers  = 8
latent_dim   = 128
n_classes    = 40
n_subjects   = 6
diff_augment_policies = "color, translation, cutout"
lr           = 3e-4
gen_lr       = 3e-4
dis_lr       = 3e-4
beta_1       = 0.5
beta_2       = 0.9999
EPOCH        = 5001 #5501#4101#2051#1024
num_col      = 32 #16#int(2 * math.log2(batch_size))
c_dim        = 512 # Conditinal Dimension Size
dis_level    = 3
feat_dim       = 128 # This will give 240 dim feature
projection_dim = 768 # projection_dim = label_dim = c_dim
input_size     = 128 # Number of EEG channels
input_shape    = (1, 440, 128)
test_batch_size= 512
generate_image = 50000
fig_freq       = 10
ckpt_freq      = 10
generate_batch_size = 1
num_layers     = 4
generate_freq  = 200
dataset_name   = 'EEGImageCVPR40'
is_cnn         = False