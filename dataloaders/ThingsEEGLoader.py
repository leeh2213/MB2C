import numpy as np
import torch

class ThingsEEG:
    def __init__(self, batch_size, batch_size_test, n_way, get_eeg_data, get_image_data):
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.args = type('', (), {})()  # Create a simple object to hold attributes
        self.args.n_way = n_way
        
        self.train_eeg, self.train_label, self.test_eeg, self.test_label = get_eeg_data()
        self.train_img_feature, self.test_center = get_image_data()

        self.prepare_data()

    def prepare_data(self):
        # Shuffle the training data
        train_shuffle = np.random.permutation(len(self.train_eeg))
        self.train_eeg = self.train_eeg[train_shuffle]
        self.train_img_feature = self.train_img_feature[train_shuffle]
        self.train_label = self.train_label[train_shuffle]

        # Split into training and validation sets
        val_eeg = torch.from_numpy(self.train_eeg[:740])
        val_image = torch.from_numpy(self.train_img_feature[:740])
        val_label = self.train_label[:740]

        self.train_eeg = torch.from_numpy(self.train_eeg[740:])
        self.train_image = torch.from_numpy(self.train_img_feature[740:])
        self.train_label = self.train_label[740:]

        # Create datasets and dataloaders
        dataset = torch.utils.data.TensorDataset(self.train_eeg, self.train_image, self.train_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        # Prepare test data
        self.test_eeg = torch.from_numpy(self.test_eeg)
        self.test_center = torch.from_numpy(self.test_center)
        self.test_label = torch.from_numpy(self.test_label)

        self.prepare_test_dataloaders()

    def prepare_test_dataloaders(self):
        # Full test dataset
        test_dataset = torch.utils.data.TensorDataset(self.test_eeg, self.test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # 50 Way test dataset
        index50 = np.squeeze(np.where(self.test_label < self.args.n_way, True, False))
        test_center_50 = self.test_center[index50]
        test_eeg_50 = self.test_eeg[index50]
        test_label_50 = self.test_label[index50]
        
        test_dataset_50 = torch.utils.data.TensorDataset(test_eeg_50, test_label_50)
        self.test_dataloader_50 = torch.utils.data.DataLoader(dataset=test_dataset_50, batch_size=self.batch_size_test, shuffle=False)