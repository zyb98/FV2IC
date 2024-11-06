import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import random


class ACDCDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = ""
        label_name = ""
        if  "train_data" in self.root_dir:
            img_name = os.path.join(self.root_dir, str(idx + 1) + '.npy')
            label_name = os.path.join("labeled_data", "train_label" + self.root_dir[-2:], str(idx + 1) + '.npy')

        if self.root_dir == "valid_data":
            img_name = os.path.join("valid_data", str(idx + 1) + '.npy')
            label_name = os.path.join("valid_label", str(idx + 1) + '.npy')

        if self.root_dir == "test_data":
            img_name = os.path.join("test_data", str(idx + 1) + '.npy')
            label_name = os.path.join("test_label", str(idx + 1) + '.npy')

        img = np.load(img_name)
        label = np.load(label_name)

        label = torch.tensor(label, dtype=torch.float32)
        label = label.view(1, 1, label.shape[0], label.shape[1])
        label1 = torch.where(label==0, torch.tensor(1), torch.tensor(0))
        label2 = torch.where(label==85, torch.tensor(1), torch.tensor(0))
        label3 = torch.where(label==170, torch.tensor(1), torch.tensor(0))
        label4 = torch.where(label==255, torch.tensor(1), torch.tensor(0))
        new_label = torch.cat((label1, label2, label3, label4))
        img = torch.tensor(img, dtype=torch.float32)

        img = img.view(1, 1, img.shape[0], img.shape[1])
        return img, new_label

def get_train_data_loader(batch_size, save_path):
    train_data = ACDCDataset(save_path)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return dataloader


class VAEDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx + 1) + '.npy')
        img = np.load(img_name)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.view(1, img.shape[0], img.shape[1])
        return img

def get_generate_data_loader(batch_size, save_path):
    train_data = VAEDataset(save_path)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return dataloader