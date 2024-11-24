from configs import cvpr40config as config
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from glob import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from encoder_modules import *
from util.CLIPModel import CLIPModel
from torchvision.models import resnet50

args = config.args
base_path       = args.base_path
train_path      = args.train_path
validation_path = args.validation_path
test_path       = args.test_path
device          = args.device
batch_size      = args.batch_size
embedding_dim   = args.embedding_dim
projection_dim  = args.projection_dim
num_layers      = args.num_layers
epochs          = args.epoch

val_log_write = open( "/data/lihao/workspace/MB2C/results/EEGCVPR40_Clip/results_val.txt", 'a+')
test_log_write = open( "/data/lihao/workspace/MB2C/results/EGCVPR40_Clip/results_test.txt", 'a+')

# load the dataset
x_train_eeg = []
x_train_image = []
labels = []

for i in tqdm(natsorted(os.listdir(base_path + train_path))):
    loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    # eeg_temp = eeg_temp / np.max(np.abs(eeg_temp))
    x_train_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_train_image.append(img)
    labels.append(loaded_array[2])

# convert to torch tensors
x_train_eeg = np.array(x_train_eeg)
x_train_image = np.array(x_train_image)
labels = np.array(labels)

x_train_eeg = torch.from_numpy(x_train_eeg).float()
x_train_image = torch.from_numpy(x_train_image).float()
labels = torch.from_numpy(labels).long()

train_data = torch.utils.data.TensorDataset(x_train_eeg, x_train_image, labels)
data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True)

### Validation dataset
x_val_eeg = []
x_val_image = []
label_Val = []

for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
    loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    # eeg_temp = eeg_temp / np.max(np.abs(eeg_temp))
    x_val_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_val_image.append(img)
    label_Val.append(loaded_array[2])

x_val_eeg = np.array(x_val_eeg)
x_val_image = np.array(x_val_image)
labels_val = np.array(label_Val)

x_val_eeg = torch.from_numpy(x_val_eeg).float().to(device)
x_val_image = torch.from_numpy(x_val_image).float().to(device)
labels_val = torch.from_numpy(labels_val).long().to(device)


### Test dataset
x_test_eeg = []
x_test_image = []
label_test = []

for i in tqdm(natsorted(os.listdir(base_path + test_path))):
    loaded_array = np.load(base_path + test_path + i, allow_pickle=True)
    eeg_temp = loaded_array[1].T
    norm     = np.max(eeg_temp)/2.0
    eeg_temp = (eeg_temp-norm)/norm
    # eeg_temp = eeg_temp / np.max(np.abs(eeg_temp))
    x_test_eeg.append(eeg_temp)
    img = cv2.resize(loaded_array[0], (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(np.transpose(img, (2, 0, 1)))/255.0
    x_test_image.append(img)
    label_test.append(loaded_array[2])

x_test_eeg = np.array(x_test_eeg)
x_test_image = np.array(x_test_image)
labels_test = np.array(label_test)

x_test_eeg = torch.from_numpy(x_test_eeg).float().to(device)
x_test_image = torch.from_numpy(x_test_image).float().to(device)
labels_test = torch.from_numpy(labels_test).long().to(device)

model_path = '/data/lihao/workspace/MB2C/model/EEGCVPR40/EXPERIMENT_15/checkpoints/'
print(natsorted(os.listdir(model_path)))

# for i in natsorted(os.listdir(model_path)):
for i in reversed(natsorted(os.listdir(model_path))):

    print("#########################################################################################")
    print('Model: ', i)
    test_log_write.write('model: %s\n' % str(i))
    val_log_write.write('model: %s\n' % str(i))

    model_path_c = model_path + i

    checkpoint = torch.load(model_path_c, map_location=device)
    eeg_embedding = EEG_Encoder(projection_dim=projection_dim, num_layers=num_layers).to(device)

    image_embedding = resnet50(pretrained=False).to(device)
    num_features = image_embedding.fc.in_features

    image_embedding.fc = nn.Sequential(
        nn.ReLU(),
        nn.Linear(num_features, args.embedding_dim, bias=False)
    )

    image_embedding.fc.to(device)

    model = CLIPModel(eeg_embedding, image_embedding, embedding_dim, projection_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    model = model.text_encoder

    for param in model.parameters():
        param.requires_grad = True

    new_layer = nn.Sequential(
        nn.Linear(args.embedding_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 40),
        nn.Softmax(dim=1)
    )

    model.fc = nn.Sequential(
        model.fc,
        new_layer
    )

    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def train(model, data_loader, optimizer, criterion, device, epoch):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs_eeg, inputs_image, labels = data
            inputs_eeg, inputs_image, labels = inputs_eeg.to(device), inputs_image.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_eeg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / i+1
        print('[%d] loss: %.3f' % (epoch + 1, loss))
        return loss

    def evaluate(model):
        model.eval()
        outputs = model(x_val_eeg)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels_val).sum().item()
        print('Accuracy Val of the network %d %%' % (100 * correct / len(labels_val)))
        val_acc = 100 * correct / len(labels_val)
        ###################################top 5######################################
        correct_top5 = 0
        top5_values, top5_indices = torch.topk(outputs.data, k=5, dim=1)
        for i in range(len(labels_val)):
            if labels_val [i] in top5_indices[i]:
                correct_top5+=1
        print('Accuracy Val_top5 of the network %d %%' % (100 * correct_top5 / len(labels_val)))
        val_acc_top5 = 100 * correct_top5 / len(labels_val)

        ###################################top 10######################################
        correct_top10 = 0
        top10_values, top10_indices = torch.topk(outputs.data, k=10, dim=1)
        for i in range(len(labels_val)):
            if labels_val [i] in top10_indices[i]:
                correct_top10+=1
        print('Accuracy Val_top10 of the network %d %%' % (100 * correct_top10 / len(labels_val)))
        val_acc_top10 = 100 * correct_top10 / len(labels_val)
        return val_acc,val_acc_top5,val_acc_top10
    
    def test(model):
        model.eval()
        outputs = model(x_test_eeg)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels_test).sum().item()
        print('Accuracy Test of the network %d %%' % (100 * correct / len(labels_test)))
        test_acc = 100 * correct / len(labels_test)

        ###################################top 5######################################
        correct_top5 = 0
        top5_values, top5_indices = torch.topk(outputs.data, k=5, dim=1)
        for i in range(len(labels_test)):
            if labels_test [i] in top5_indices[i]:
                correct_top5+=1
        print('Accuracy Test_top5 of the network %d %%' % (100 * correct_top5 / len(labels_test)))
        test_acc_top5 = 100 * correct_top5 / len(labels_test)

        ###################################top 10######################################
        correct_top10 = 0
        top10_values, top10_indices = torch.topk(outputs.data, k=10, dim=1)
        for i in range(len(labels_test)):
            if labels_test [i] in top10_indices[i]:
                correct_top10+=1
        print('Accuracy Test_top10 of the network %d %%' % (100 * correct_top10 / len(labels_test)))
        test_acc_top10 = 100 * correct_top10 / len(labels_test)
        return test_acc,test_acc_top5,test_acc_top10


    dir_info  = natsorted(glob('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1

    if not os.path.isdir('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}'.format(experiment_num)):
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}'.format(experiment_num))
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/val/tsne'.format(experiment_num))
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/train/tsne/'.format(experiment_num))
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/test/tsne/'.format(experiment_num))
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/test/umap/'.format(experiment_num))
        os.system('cp *.py /data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}'.format(experiment_num))

    ckpt_lst = natsorted(glob('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

    START_EPOCH = 0

    if len(ckpt_lst)>=1:
        ckpt_path  = ckpt_lst[-1]
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
        START_EPOCH += 1
    else:
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/checkpoints/'.format(experiment_num))
        os.makedirs('/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/bestckpt/'.format(experiment_num))

    best_val_acc   = 0.0
    best_val_acc_top5   = 0.0
    best_val_epoch = 0

    for epoch in range(START_EPOCH, epochs):

        running_train_loss = train(model, data_loader, optimizer, criterion, device, epoch)
        val_acc,val_acc_top5,val_acc_top10 = evaluate(model)
        val_log_write.write('Epoch %d: , top1: %.4f ,top5: %.4f ,top10: %.4f\n'%((epoch + 1), val_acc ,val_acc_top5, val_acc_top10))



        if best_val_acc <= val_acc or best_val_acc_top5<=val_acc_top5:
            test_acc,test_acc_top5,test_acc_top10 = test(model)
            test_log_write.write('Epoch %d: , top1: %.4f ,top5: %.4f ,top10: %.4f\n'%((epoch + 1), test_acc ,test_acc_top5,test_acc_top10))
            best_val_acc   = val_acc
            best_val_acc_top5   = val_acc_top5
            best_val_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, '/data/lihao/workspace/MB2C/results/FineTuningEEG_CVPR40/EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, best_val_epoch, val_acc))
    del model
    torch.cuda.empty_cache()
    gc.collect()
