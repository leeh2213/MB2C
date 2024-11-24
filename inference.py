import os
import argparse
import random
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.end_to_end_encoding_utils import *
from EEG_Encoder.SelfModel import *
from encoder_modules import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA



gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = 'results/' 
model_idx = 'stage2_encoder_training_from_scratch_clip'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with vit encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='1000', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training.')
parser.add_argument('--reproduce', type=bool, default=True)
parser.add_argument('--n_way', type=int, default=200)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--is_gan', type=bool, default=True)


 
class Infer():
    def __init__(self, args, nsub, nseed):
        super(Infer, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.nseed = nseed

        self.start_epoch = 0
        self.eeg_data_path = 'Data/Things-EEG2/Preprocessed_data_250Hz/'
        self.img_data_path = 'Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/' + args.dnn + '/pretrained-True/'
        self.test_center_path = 'Data/Things-EEG2/Image_set/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, 'a+')
        self.writer = SummaryWriter(log_dir=result_path+'/log/')

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.Enc_eeg = Enc_eeg().to(args.device)
        self.Proj_eeg = Proj_eeg().to(args.device)
        self.Proj_img = Proj_img().to(args.device)
        
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.Enc_eeg.load_state_dict(torch.load('model/best_checkpoints_seed42/' + model_idx + str(self.nSub) + 'Enc_eeg_cls.pth'))
        self.Proj_eeg.load_state_dict(torch.load('model/best_checkpoints_seed42/' + model_idx  +str(self.nSub) + 'Proj_eeg_cls.pth'))
        self.Proj_img.load_state_dict(torch.load('model/best_checkpoints_seed42/' + model_idx +str(self.nSub) + 'Proj_img_cls.pth'))

        self.centers = {}
        print('initial define done.')


    def get_eeg_data(self):
        train_data = []
        train_label = torch.from_numpy(np.repeat(np.arange(1654), 10))
        test_data = []
        test_label = np.arange(200)
        
        train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        # (16540, 4, 63, 250)
        train_data = train_data['preprocessed_eeg_data'] 
        train_data = np.mean(train_data, axis=1)
        # (16540, 1, 63, 250)
        train_data = np.expand_dims(train_data, axis=1)
        
        # (200, 80, 63, 250)
        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        test_data = np.mean(test_data, axis=1)
        # (200, 1, 63, 250)
        test_data = np.expand_dims(test_data, axis=1)
        # train_label:[] test_label: 0~199 
        print('load eeg successful')
        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)
        test_img_feature = np.squeeze(test_img_feature)
        print('load imag successful')
        return test_img_feature

    def InferenceAllSubs(self, args):
        # train_eeg:(16540, 1, 63, 250); test_eeg:(200, 1, 63, 250); test_label：0~199
        train_eeg_data, train_eeg_label, test_eeg, test_label = self.get_eeg_data()
        train_eeg_data = torch.from_numpy(train_eeg_data)
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        train_img_feature = torch.from_numpy(np.squeeze(train_img_feature))
        
        train_dataset = torch.utils.data.TensorDataset(train_eeg_data, train_eeg_label)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        # (16540, 768)
        test_center = self.get_image_data()
        
        test_center = np.load(self.test_center_path + 'center_all_image_' + self.args.dnn + '.npy', allow_pickle=True)
        # (200, 768)
        test_center = np.squeeze(test_center,1)

        #####################################200 Way################################
        test_eeg = torch.from_numpy(test_eeg)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)
        
        #####################################50 Way################################
        index50 = np.squeeze(np.where(test_label < self.args.n_way, True, False))
        test_center_50 = test_center[index50]
        test_eeg_50 = test_eeg[index50]
        test_label_50 = test_label[index50]
        test_dataset_50 = torch.utils.data.TensorDataset(test_eeg_50, test_label_50)
        self.test_dataloader_50 = torch.utils.data.DataLoader(dataset=test_dataset_50, batch_size=self.batch_size_test, shuffle=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():            
            eeg_feat = self.Proj_eeg(self.Enc_eeg(Variable(test_eeg.type(self.Tensor))))
            proj_test_center = self.Proj_img(test_center)
            norm_test_center = proj_test_center / proj_test_center.norm(dim=1, keepdim=True)
            norm_test_eeg_feat = eeg_feat / eeg_feat.norm(dim=1, keepdim=True)

            gt_labels, predict_labels = get_inference_label(norm_test_center, norm_test_eeg_feat, test_label)
            top1_acc, top3_acc, top5_acc = run_classification_test(args, norm_test_center, eeg_feat, 0)

            eeg_feat_50 = self.Proj_eeg(self.Enc_eeg(Variable(test_eeg_50.type(self.Tensor))))
            proj_test_center_50 = self.Proj_img(test_center_50)
            norm_eeg_feat_50 = eeg_feat_50 / eeg_feat_50.norm(dim=1, keepdim=True)
            norm_test_center_50 = proj_test_center_50 / proj_test_center_50.norm(dim=1, keepdim=True)
            top1_acc_50, top3_acc_50, top5_acc_50 = run_classification_test(args, norm_test_center_50, norm_eeg_feat_50, 0)

        print('The test Top1_200-%.6f, Top3_200-%.6f, Top5_200-%.6f' % (top1_acc, top3_acc, top5_acc))
        print('The test Top1_50-%.6f, Top3_50-%.6f, Top5_50-%.6f' % (top1_acc_50, top3_acc_50, top5_acc_50))

        self.log_write.write('The test Top1_200-%.6f, Top3_200-%.6f, Top5_200-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The test Top1_50-%.6f, Top3_50-%.6f, Top5_50-%.6f\n' % (top1_acc_50, top3_acc_50, top5_acc_50))

        self.writer.close()
        return top1_acc, top3_acc, top5_acc, top1_acc_50, top3_acc_50, top5_acc_50

def main():
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_sub = args.num_sub   
    cal_num = 0
    for j in range(1):
        aver = []
        aver3 = []
        aver5 = []
        aver_50 = []
        aver3_50 = []
        aver5_50 = []
        for i in range(10):
            cal_num += 1
            print('Subject %d' % (i+1))
            infer = Infer(args, i + 1,j+1)
            starttime = datetime.datetime.now()

            Acc, Acc3, Acc5, Acc_50, Acc3_50, Acc5_50 = infer.InferenceAllSubs(args)
            print('THE BEST ACCURACY IS ' + str(Acc))


            endtime = datetime.datetime.now()
            print('subject %d duration: '%(i+1) + str(endtime - starttime))

            aver.append(Acc)
            aver3.append(Acc3)
            aver5.append(Acc5)
            aver_50.append(Acc_50)
            aver3_50.append(Acc3_50)
            aver5_50.append(Acc5_50)
            
        aver.append(np.mean(aver))
        aver3.append(np.mean(aver3))
        aver5.append(np.mean(aver5))
        aver_50.append(np.mean(aver_50))
        aver3_50.append(np.mean(aver3_50))
        aver5_50.append(np.mean(aver5_50))

        column = np.arange(1, cal_num+1).tolist()
        column.append('ave')
        pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5, aver_50, aver3_50, aver5_50])
        pd_all.to_csv(result_path + 'result.csv', mode='a',)
        
if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))