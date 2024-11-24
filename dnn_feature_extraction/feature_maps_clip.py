"""
Package all the CLIP features

"""

import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/data/lihao/workspace/MB2C/Data/Things-EEG2/', type=str)
parser.add_argument('--MixRatio', type=float, default=0.25)
args = parser.parse_args()
def get_eeg_data():
    data = []
    for i in range(1,11):
        train_data = np.load('/data/lihao/workspace/MB2C/Data/Things-EEG2/Preprocessed_data_250Hz/' + '/sub-' + format(i, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
		# (16540, 4, 63, 250)
        train_data = train_data['preprocessed_eeg_data'] 
        train_data = np.mean(train_data, axis=1)
		# (16540, 1, 63, 250)
        train_data = np.expand_dims(train_data, axis=1)
        data.append(train_data)
        print('success load eeg sub',i)

    return data

def augment_fuc(args, image_data, eeg_data, labels):
    index1 = np.random.permutation(len(image_data))
    index2 = np.random.permutation(len(image_data))
    batch_size = int(len(image_data) * args.MixRatio)
    mix_image, mix_image1, mix_image2, mix_image3= [], [], [], []
    labels_a, labels_b = [], []
    lam = 0.5
    '''Returns mixed inputs, pairs of targets, and lambda'''
    alpha = 1.0
    for i in tqdm(range(batch_size)):
    # for i in range(5):
        # lam = np.random.beta(alpha, alpha)
        # if lam < 0.4 or lam > 0.6:
        #     lam = 0.5
        if i < int(batch_size*0.25):
            if labels[index1[i]]!=labels[index2[i]]:
                if len(mix_image)==0:
                    mix_image = (lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])
                    mix_eeg1 = (lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])
                    mix_eeg2 = (lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])
                    mix_eeg3 = (lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])
                    mix_eeg4 = (lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])
                    mix_eeg5 = (lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])
                    mix_eeg6 = (lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])
                    mix_eeg7 = (lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])
                    mix_eeg8 = (lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])
                    mix_eeg9 = (lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])
                    mix_eeg10 = (lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])
                    
                    labels_a = labels[index1[i]].unsqueeze(0)
                    labels_b = labels[index2[i]].unsqueeze(0)
                    # lams.append(lam)
                else:
                    # image mixup
                    mix_image = np.concatenate((mix_image,(lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])), axis=0)
                    mix_eeg1 = np.concatenate((mix_eeg1,(lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])), axis=0)
                    mix_eeg2 = np.concatenate((mix_eeg2,(lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])), axis=0)
                    mix_eeg3 = np.concatenate((mix_eeg3,(lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])), axis=0)
                    mix_eeg4 = np.concatenate((mix_eeg4,(lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])), axis=0)
                    mix_eeg5 = np.concatenate((mix_eeg5,(lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])), axis=0)
                    mix_eeg6 = np.concatenate((mix_eeg6,(lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])), axis=0)
                    mix_eeg7 = np.concatenate((mix_eeg7,(lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])), axis=0)
                    mix_eeg8 = np.concatenate((mix_eeg8,(lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])), axis=0)
                    mix_eeg9 = np.concatenate((mix_eeg9,(lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])), axis=0)
                    mix_eeg10 = np.concatenate((mix_eeg10,(lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])), axis=0)
                    labels_a = np.concatenate((labels_a,labels[index1[i]].unsqueeze(0)), axis=0)
                    labels_b = np.concatenate((labels_b,labels[index2[i]].unsqueeze(0)), axis=0)
                    # lams.append(lam)
        elif i < int(batch_size*0.5):
            if len(mix_image1)==0:
                mix_image1 = (lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])
                mix_eeg11 = (lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])
                mix_eeg22 = (lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])
                mix_eeg33 = (lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])
                mix_eeg44 = (lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])
                mix_eeg55 = (lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])
                mix_eeg66 = (lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])
                mix_eeg77 = (lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])
                mix_eeg88 = (lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])
                mix_eeg99 = (lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])
                mix_eeg1010 = (lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])
                
                labels_a1 = labels[index1[i]].unsqueeze(0)
                labels_b1 = labels[index2[i]].unsqueeze(0)
                # lams.append(lam)    
            # image mixup
            mix_image1 = np.concatenate((mix_image1,(lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])), axis=0)
            mix_eeg11 = np.concatenate((mix_eeg11,(lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])), axis=0)
            mix_eeg22 = np.concatenate((mix_eeg22,(lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])), axis=0)
            mix_eeg33 = np.concatenate((mix_eeg33,(lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])), axis=0)
            mix_eeg44 = np.concatenate((mix_eeg44,(lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])), axis=0)
            mix_eeg55 = np.concatenate((mix_eeg55,(lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])), axis=0)
            mix_eeg66 = np.concatenate((mix_eeg66,(lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])), axis=0)
            mix_eeg77 = np.concatenate((mix_eeg77,(lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])), axis=0)
            mix_eeg88 = np.concatenate((mix_eeg88,(lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])), axis=0)
            mix_eeg99 = np.concatenate((mix_eeg99,(lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])), axis=0)
            mix_eeg1010 = np.concatenate((mix_eeg1010,(lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])), axis=0)
            labels_a1 = np.concatenate((labels_a1,labels[index1[i]].unsqueeze(0)), axis=0)
            labels_b1 = np.concatenate((labels_b1,labels[index2[i]].unsqueeze(0)), axis=0)
            # lams.append(lam)
        elif i < int(batch_size*0.75): 
            if len(mix_image2)==0:
                mix_image2 = (lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])
                mix_eeg111 = (lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])
                mix_eeg222 = (lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])
                mix_eeg333 = (lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])
                mix_eeg444 = (lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])
                mix_eeg555 = (lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])
                mix_eeg666 = (lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])
                mix_eeg777 = (lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])
                mix_eeg888 = (lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])
                mix_eeg999 = (lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])
                mix_eeg101010 = (lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])
                
                labels_a11 = labels[index1[i]].unsqueeze(0)
                labels_b11 = labels[index2[i]].unsqueeze(0)
                # lams.append(lam)      
            # image mixup
            mix_image2 = np.concatenate((mix_image2,(lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])), axis=0)
            mix_eeg111 = np.concatenate((mix_eeg111,(lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])), axis=0)
            mix_eeg222 = np.concatenate((mix_eeg222,(lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])), axis=0)
            mix_eeg333 = np.concatenate((mix_eeg333,(lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])), axis=0)
            mix_eeg444 = np.concatenate((mix_eeg444,(lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])), axis=0)
            mix_eeg555 = np.concatenate((mix_eeg555,(lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])), axis=0)
            mix_eeg666 = np.concatenate((mix_eeg666,(lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])), axis=0)
            mix_eeg777 = np.concatenate((mix_eeg777,(lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])), axis=0)
            mix_eeg888 = np.concatenate((mix_eeg888,(lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])), axis=0)
            mix_eeg999 = np.concatenate((mix_eeg999,(lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])), axis=0)
            mix_eeg101010 = np.concatenate((mix_eeg101010,(lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])), axis=0)
            labels_a11 = np.concatenate((labels_a11,labels[index1[i]].unsqueeze(0)), axis=0)
            labels_b11 = np.concatenate((labels_b11,labels[index2[i]].unsqueeze(0)), axis=0)
            # lams.append(lam)
        else:    
            if len(mix_image3)==0:
                mix_image3 = (lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])
                mix_eeg1111 = (lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])
                mix_eeg2222 = (lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])
                mix_eeg3333 = (lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])
                mix_eeg4444 = (lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])
                mix_eeg5555 = (lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])
                mix_eeg6666 = (lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])
                mix_eeg7777 = (lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])
                mix_eeg8888 = (lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])
                mix_eeg9999 = (lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])
                mix_eeg10101010 = (lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])
                
                labels_a111 = labels[index1[i]].unsqueeze(0)
                labels_b111 = labels[index2[i]].unsqueeze(0)
                # lams.append(lam)   
            # image mixup
            mix_image3 = np.concatenate((mix_image3,(lam * image_data[index1[i]] + (1 - lam) * image_data[index2[i]])), axis=0)
            mix_eeg1111 = np.concatenate((mix_eeg1111,(lam * eeg_data[0][index1[i]] + (1 - lam) * eeg_data[0][index2[i]])), axis=0)
            mix_eeg2222 = np.concatenate((mix_eeg2222,(lam * eeg_data[1][index1[i]] + (1 - lam) * eeg_data[1][index2[i]])), axis=0)
            mix_eeg3333 = np.concatenate((mix_eeg3333,(lam * eeg_data[2][index1[i]] + (1 - lam) * eeg_data[2][index2[i]])), axis=0)
            mix_eeg4444 = np.concatenate((mix_eeg4444,(lam * eeg_data[3][index1[i]] + (1 - lam) * eeg_data[3][index2[i]])), axis=0)
            mix_eeg5555 = np.concatenate((mix_eeg5555,(lam * eeg_data[4][index1[i]] + (1 - lam) * eeg_data[4][index2[i]])), axis=0)
            mix_eeg6666 = np.concatenate((mix_eeg6666,(lam * eeg_data[5][index1[i]] + (1 - lam) * eeg_data[5][index2[i]])), axis=0)
            mix_eeg7777 = np.concatenate((mix_eeg7777,(lam * eeg_data[6][index1[i]] + (1 - lam) * eeg_data[6][index2[i]])), axis=0)
            mix_eeg8888 = np.concatenate((mix_eeg8888,(lam * eeg_data[7][index1[i]] + (1 - lam) * eeg_data[7][index2[i]])), axis=0)
            mix_eeg9999 = np.concatenate((mix_eeg9999,(lam * eeg_data[8][index1[i]] + (1 - lam) * eeg_data[8][index2[i]])), axis=0)
            mix_eeg10101010 = np.concatenate((mix_eeg10101010,(lam * eeg_data[9][index1[i]] + (1 - lam) * eeg_data[9][index2[i]])), axis=0)
            labels_a111 = np.concatenate((labels_a111,labels[index1[i]].unsqueeze(0)), axis=0)
            labels_b111 = np.concatenate((labels_b111,labels[index2[i]].unsqueeze(0)), axis=0)
            # lams.append(lam)
        # else:
        #     batch_size += 1
    print('success augment')
    # 创建一个形状为（1654，2）的新数组，初始化为0
    expanded_labels = np.zeros((labels.shape[0], 2), dtype=int)
    # 将原始标签放入新数组的第一列
    expanded_labels[:, 0] = labels
    # 将新数组的第二列的所有元素设置为-1
    expanded_labels[:, 1] = -1
    
    aug_label = np.concatenate((labels_a[:, np.newaxis],labels_b[:, np.newaxis]), axis=1)
    aug_label1 = np.concatenate((labels_a1[:, np.newaxis],labels_b1[:, np.newaxis]), axis=1)
    aug_label2 = np.concatenate((labels_a11[:, np.newaxis],labels_b11[:, np.newaxis]), axis=1)
    aug_label3 = np.concatenate((labels_a111[:, np.newaxis],labels_b111[:, np.newaxis]), axis=1)


    labels = np.concatenate((expanded_labels,aug_label,aug_label1,aug_label2,aug_label3), axis=0)
    image_data = np.array(image_data)
    # (_,768)
    image = np.concatenate((image_data.squeeze(1),mix_image,mix_image1,mix_image2,mix_image3), axis=0)
    # (_,1,63,250)
    eeg1 = np.concatenate((eeg_data[0], mix_eeg1[:, np.newaxis,:,:], mix_eeg11[:, np.newaxis,:,:], mix_eeg111[:, np.newaxis,:,:], mix_eeg1111[:, np.newaxis,:,:]), axis=0)
    eeg2 = np.concatenate((eeg_data[1], mix_eeg2[:, np.newaxis,:,:], mix_eeg22[:, np.newaxis,:,:], mix_eeg222[:, np.newaxis,:,:], mix_eeg2222[:, np.newaxis,:,:]), axis=0)
    eeg3 = np.concatenate((eeg_data[2], mix_eeg3[:, np.newaxis,:,:], mix_eeg33[:, np.newaxis,:,:], mix_eeg333[:, np.newaxis,:,:], mix_eeg3333[:, np.newaxis,:,:]), axis=0)
    eeg4 = np.concatenate((eeg_data[3], mix_eeg4[:, np.newaxis,:,:], mix_eeg44[:, np.newaxis,:,:], mix_eeg444[:, np.newaxis,:,:], mix_eeg4444[:, np.newaxis,:,:]), axis=0)
    eeg5 = np.concatenate((eeg_data[4], mix_eeg5[:, np.newaxis,:,:], mix_eeg55[:, np.newaxis,:,:], mix_eeg555[:, np.newaxis,:,:], mix_eeg5555[:, np.newaxis,:,:]), axis=0)
    eeg6 = np.concatenate((eeg_data[5], mix_eeg6[:, np.newaxis,:,:], mix_eeg66[:, np.newaxis,:,:], mix_eeg666[:, np.newaxis,:,:], mix_eeg6666[:, np.newaxis,:,:]), axis=0)
    eeg7 = np.concatenate((eeg_data[6], mix_eeg7[:, np.newaxis,:,:], mix_eeg77[:, np.newaxis,:,:], mix_eeg777[:, np.newaxis,:,:], mix_eeg7777[:, np.newaxis,:,:]), axis=0)
    eeg8 = np.concatenate((eeg_data[7], mix_eeg8[:, np.newaxis,:,:], mix_eeg88[:, np.newaxis,:,:], mix_eeg888[:, np.newaxis,:,:], mix_eeg8888[:, np.newaxis,:,:]), axis=0)
    eeg9 = np.concatenate((eeg_data[8], mix_eeg9[:, np.newaxis,:,:], mix_eeg99[:, np.newaxis,:,:], mix_eeg999[:, np.newaxis,:,:], mix_eeg9999[:, np.newaxis,:,:]), axis=0)
    eeg10 = np.concatenate((eeg_data[9], mix_eeg10[:, np.newaxis,:,:], mix_eeg1010[:, np.newaxis,:,:], mix_eeg101010[:, np.newaxis,:,:], mix_eeg10101010[:, np.newaxis,:,:]), axis=0)
    mix_eeg_list = [eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, eeg8, eeg9, eeg10]

    # return image, mix_eeg_list, labels, lams
    return image, mix_eeg_list, labels


print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Load the feature maps
feats = []
fmaps_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'training_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps))
	feats.append(fmaps_data)

labels = torch.from_numpy(np.repeat(np.arange(1654), 10))
eeg = get_eeg_data()
mix_image, mix_eeg_list, labels= augment_fuc(args, feats, eeg, labels)
# lams = np.array(lams)

# Save image
save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
	'aug_pca_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained))
file_name = 'clip_feature_maps_training'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), mix_image)
del feats

# Save eeg
for i in range(1,11):
    eeg_data_path = '/data/lihao/workspace/MB2C/Data/TData//Things-EEG2//Aug_Preprocessed_data_250Hz/' + '/sub-' + format(i, '02') + '/'
    if os.path.isdir(eeg_data_path) == False:
        os.makedirs(eeg_data_path)
    file_name = 'preprocessed_eeg_training'
    np.save(os.path.join(eeg_data_path, file_name), mix_eeg_list[i-1])
    
np.save(os.path.join('/data/lihao/workspace/MB2C/Data/TData//Things-EEG2//Aug_Preprocessed_data_250Hz//', 'labels_'+str(args.MixRatio)), labels)
# np.save(os.path.join('/data/lihao/workspace/MB2C/Data/TData//Things-EEG2//Aug_Preprocessed_data_250Hz//', 'lams_'+str(args.MixRatio)), lams)

print('mix_image shape:',mix_image.shape)
print('mix_eeg_list[0] shape:',mix_eeg_list[0] .shape)
print('labels shape:',labels.shape)

# Load the feature maps
# feats = []
# fmaps_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
# 	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
# 	'test_images')
# fmaps_list = os.listdir(fmaps_dir)
# fmaps_list.sort()
# for f, fmaps in enumerate(fmaps_list):
# 	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps))
# 	feats.append(fmaps_data)

# # Save the downsampled feature maps
# file_name = 'clip_feature_maps_test'
# np.save(os.path.join(save_dir, file_name), feats)
# del feats

