# from model.EEGModel import *
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience  # 容忍多少个epoch没有性能提升
        self.verbose = verbose
        self.counter = 0  # 记录连续性能没有提升的epoch数
        self.best_score = None
        self.early_stop = False  # 是否停止训练

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss >= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

        return self.early_stop

def load_images(args, idx_val, image_model):
    """Load and preprocess the training, validation and test images.

    Parameters
    ----------
    args : Namespace
            Input arguments.
    idx_val : bool
            Indices of the validation images.

    Returns
    -------
    X_train : list of tensor
            Training images.
    X_val : list of tensor
            Validation images.
    X_test : list of tensor
            Test images.

    """

    import os
    from torchvision import transforms
    from tqdm import tqdm
    from PIL import Image

    if args.image_model == 'CLIP':
        preprocess = image_model.preprocess
    ### Define the image preprocesing ###
    else:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    ### Load and preprocess the training and validation images ###
    X_train = []
    X_val = []
    if not args.reproduce:
        img_dirs = os.path.join(args.project_dir, 'image_set', 'training_images')
        image_list = []
        for root, dirs, files in os.walk(img_dirs):
            for file in files:
                if file.endswith(".jpg"):
                    image_list.append(os.path.join(root, file))
        image_list.sort()
        for i, image in enumerate(tqdm(image_list)):
            img = Image.open(image).convert('RGB')
            img = preprocess(img)
            if idx_val[i] == True:
                X_val.append(img)
            else:
                X_train.append(img)
        X_train = torch.stack(X_train, dim=0)
        X_val = torch.stack(X_val, dim=0)

    ### Load and preprocess the test images ###
    img_dirs = os.path.join(args.project_dir, 'image_set', 'test_images')
    image_list = []
    for root, dirs, files in os.walk(img_dirs):
        for file in files:
            if file.endswith(".jpg"):
                image_list.append(os.path.join(root, file))
    image_list.sort()
    X_test = []
    for image in tqdm(image_list):
        img = Image.open(image).convert('RGB')
        img = preprocess(img)
        X_test.append(img)

    X_test = torch.stack(X_test, dim=0)

    ### Output ###
    return X_train, X_val, X_test

def load_eeg_data(args, idx_val):
    """Load the EEG training and test data.

    Parameters
    ----------
    args : Namespace
            Input arguments.
    idx_val : bool
            Indices of the validation images.

    Returns
    -------
    y_train : tensor
            Training EEG data.
    y_val : tensor
            Validation EEG data.
    y_test : tensor
            Test EEG data.
    ch_names : list of str
            EEG channel names.
    times : float
            EEG time points.

    """
    ### Load the EEG training data ###
    y_train, y_val, y_test=[],[],[]
    data_dir = os.path.join('eeg_dataset', 'preprocessed_mean_data', 'sub-' +
                            format(args.sub, '02'))
    data_dir_test = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-' +
                            format(args.sub, '02'))
    training_file = 'preprocessed_eeg_training.npy'
    test_file = 'preprocessed_eeg_test.npy'
    data = np.load(os.path.join(args.project_dir, data_dir_test, test_file),
                allow_pickle=True).item()
    # 200, 80, 63, 250
    y_test = data['preprocessed_eeg_data'][:, :, :, 50:]
    # y_test = torch.from_numpy(y_test).permute(1, 0, 2, 3).contiguous().view(-1, 63, 250)
    y_test = torch.tensor(np.float32(y_test))
    data = np.load(os.path.join(args.project_dir, data_dir, training_file),
                allow_pickle=True).item()
    # data = joblib.load(os.path.join(args.project_dir, data_dir, training_file))
    y_train = data['preprocessed_eeg_data'][:, :, 50:]
    # Average across repetitions
    # y_train = np.mean(y_train, 1)
    # Extract the validation data
    y_val = y_train[idx_val]
    y_train = np.delete(y_train, idx_val, 0)
    # Convert to float32 and tensor (for DNN training with Pytorch)
    y_train = torch.tensor(np.float32(y_train))
    y_val = torch.tensor(np.float32(y_val))
    # Average across repetitions
    # Convert to float32 and tensor (for DNN training with Pytorch)
    y_test = torch.tensor(np.float32(y_test))
    return y_train, y_val, y_test

def create_dataloader(args, Image_train, Image_val, EEG_train, EEG_val):
    """Put the training, validation and test data into a PyTorch-compatible
    Dataloader format.

    Parameters
    ----------
    args : Namespace
            Input arguments.
    time_point : int
            Modeled EEG time point.
    g_cpu : torch.Generator
            Generator object for DataLoader random batching.
    Image_train : list of tensor
            Training images.
    Image_val : list of tensor
            Validation images.
    Image_test : list of tensor
            Test images.
    EEG_train : float
            Training EEG data.
    EEG_train : float
            Validation EEG data.
    EEG_test : float
            Test EEG data.

    Returns
    ----------
    train_dl : Dataloader
            Training Dataloader.
    val_dl : Dataloader
            Validation Dataloader.
    test_dl : Dataloader
            Test Dataloader.

    """

    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

    ### Dataset class ###
    class EegDataset(Dataset):
        def __init__(self, image, eeg):
            self.image = image
            self.eeg = eeg
            assert self.image.size(0) == self.eeg.size(0)

        def __len__(self):
            return self.eeg.size(0)

        def __getitem__(self, idx):
            return self.image[idx], self.eeg[idx]

    ### Convert the data to PyTorch's Dataset format ###
    train_ds = EegDataset(Image_train, EEG_train)
    val_ds = EegDataset(Image_val, EEG_val)
#     test_ds = EegDataset(Image_test, EEG_test)

    ### Convert the Datasets to PyTorch's Dataloader format ###
    train_dl = DataLoader(
        dataset=train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_dl = DataLoader(
        dataset=val_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
#     test_dl = DataLoader(dataset=test_ds, batch_size=test_ds.__len__(),shuffle=False, pin_memory=True)
#     test_dl = DataLoader(
#         dataset=test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    ### Output ###
    return train_dl, val_dl

def preprocess_image(data, num_times):
    # data 是包含 1000 个 3x224x224 张量的列表
    # 将列表中的张量叠加成一个更大的张量
    stacked = torch.stack(data)  # 形状：(, 3, 224, 224)
    # 对这个大张量进行重复
    repeated_image = stacked.unsqueeze(1).repeat(
        1, num_times, 1, 1, 1)  # 形状：(, 10, 3, 224, 224)

    return repeated_image

def preprocess_eeg(data, num_times):
    trails, channels, timepoints = data.shape[0], data.shape[1], data.shape[1]
    split_eeg = data.view(trails, channels, num_times, -1)
    final_eeg = split_eeg.permute(0, 2, 1, 3)

    return final_eeg

def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    return optimizer

def save_checkpoint(state, logdir, filename='checkpoint.pt'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    path = os.path.join(logdir, filename)
    torch.save(state, path)

def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler
    from logging import Formatter, DEBUG, ERROR, INFO  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(INFO)
    sh = StreamHandler()
    sh.setLevel(INFO)

    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_inference_label(image_test,EEG_test,labels_test):
    # 对每个图像进行零样本分类
    top1acc = 0
    ground_truth_labels = []
    predict_labels = []
    top5_retrival = {}
    sum_samples = len(labels_test)
    
    # 计算EEG_test和image_test之间的相似度矩阵
    labels_test = labels_test.cpu().numpy()
    eeg_embeddings = EEG_test.cpu().numpy()
    image_embeddings = image_test.cpu().numpy()
    similarity_matrix = np.dot(eeg_embeddings, image_embeddings.T)

    # 计算准确率
    for i in range(sum_samples):
        # 计算 top-1 准确率
        ground_truth_labels.append(labels_test[i])
        predict_labels.append(labels_test[np.argmax(similarity_matrix[i])])
        if labels_test[i]==labels_test[np.argmax(similarity_matrix[i])]:
            top1acc += 1
        top5_indices = np.argsort(similarity_matrix[i])[-5:]
        top3_indices = np.argsort(similarity_matrix[i])[-3:]
        if (labels_test[i] in labels_test[top5_indices]) and (labels_test[i] in labels_test[top3_indices]):
            top5_retrival[labels_test[i]] = labels_test[top5_indices][::-1].tolist()
            
    # 打开文件以追加的方式
    with open("pic/output.txt", "a") as file:
        file.write(repr(predict_labels) + "\n")
        file.close()
    with open("pic/outputTOP5.txt", "a") as file:
        file.write(repr(top5_retrival) + "\n")
        file.close()
    return ground_truth_labels, predict_labels
    
def run_classification_test(args,image_test,EEG_test,epoch):
    # 对每个图像进行零样本分类
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    image_embeddings = image_test
    sum_samples = EEG_test.shape[0]
    # if args.reproduce:
    eeg_embeddings = EEG_test
    # 先将 CUDA 张量移动到 CPU
    eeg_embeddings = eeg_embeddings.cpu().numpy()
    image_embeddings = image_embeddings.cpu().numpy()
    similarity_matrix = np.dot(eeg_embeddings, image_embeddings.T)
    # 找到每一行的最大值
    max_values = np.max(similarity_matrix, axis=1, keepdims=True)
    # 将每一行的最大值设置为 1，其他值设置为 0
    normalized_matrix1 = np.where(similarity_matrix == max_values, 1, 0)
    # 对角线元素和,即预测正确的数量
    top1_correct += np.trace(normalized_matrix1)

    # 找到每一行的前3个最大值的索引
    top3_indices = np.argsort(similarity_matrix, axis=1)[:, -3:]
    # 创建一个与相似度矩阵相同大小的矩阵，初始值为0
    normalized_matrix3 = np.zeros_like(similarity_matrix)
    # 将每一行的前3个最大值对应的位置设置为1
    normalized_matrix3[np.arange(normalized_matrix3.shape[0])[:, None], top3_indices] = 1
    top3_correct += np.trace(normalized_matrix3)

    # 找到每一行的前5个最大值的索引
    top5_indices = np.argsort(similarity_matrix, axis=1)[:, -5:]
    # 创建一个与相似度矩阵相同大小的矩阵，初始值为0
    normalized_matrix5 = np.zeros_like(similarity_matrix)
    # 将每一行的前5个最大值对应的位置设置为1
    normalized_matrix5[np.arange(normalized_matrix5.shape[0])[:, None], top5_indices] = 1
    top5_correct += np.trace(normalized_matrix5)
    # 计算准确率
    top1 = top1_correct / sum_samples
    top3 = top3_correct / sum_samples
    top5= top5_correct / sum_samples
    print("Test ACC Epoch:{} top1:{} top3:{} top5:{}".format(epoch, top1, top3, top5))
    return top1, top3, top5

def run_classification_test_labels(EEG_test, image_test, labels, epoch):
    # 对每个图像进行零样本分类
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    sum_samples = len(labels)
    
    # 计算EEG_test和image_test之间的相似度矩阵
    eeg_embeddings = EEG_test.cpu().numpy()
    image_embeddings = image_test.cpu().numpy()
    similarity_matrix = np.dot(eeg_embeddings, image_embeddings.T)

    # 计算准确率
    for i in range(sum_samples):
        # 计算 top-1 准确率
        if labels[np.argmax(similarity_matrix[i])] == labels[i]:
            top1_correct += 1
        
        # 计算 top-3 准确率
        top3_indices = np.argsort(similarity_matrix[i])[-3:]
        if labels[i] in labels[top3_indices]:
            top3_correct += 1
        
        # 计算 top-5 准确率
        top5_indices = np.argsort(similarity_matrix[i])[-5:]
        if labels[i] in labels[top5_indices]:
            top5_correct += 1
            
    # 计算准确率
    top1 = top1_correct / sum_samples
    top3 = top3_correct / sum_samples
    top5 = top5_correct / sum_samples
    
    # 输出结果
    # print("Test ACC Epoch:{} top1:{} top3:{} top5:{}".format(epoch, top1, top3, top5))
    
    return top1, top3, top5

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1)), dtype=torch.float32, device='cuda')
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.tensor(np.ones(d_interpolates.shape), dtype=torch.float32, device='cuda'))
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # gradients = gradients.view(gradients.size(0), -1)
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


    EEG_test=EEG_test.to(args.device)
    with torch.no_grad():
        EEG_test=EEG_test.float()
        eeg_model.eval()
        results = eeg_model(EEG_test)
        return results.cpu().numpy()