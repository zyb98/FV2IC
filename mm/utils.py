import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import random


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        # smooth = 0.1
        # probs = F.softmax(logits, dim=2)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2).sum()
        score = (2 * intersection ) / (m1.sum() + m2.sum())
        score = 1 - score / num
        return score

class ACDCDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = ""
        label_name = ""
        if  "train_data" in self.root_dir:
            # read image and labels
            img_name = os.path.join(self.root_dir, str(idx + 1) + '.npy')
            label_name = os.path.join("new_upper", "train_label" + self.root_dir[-2:], str(idx + 1) + '.npy')
        if self.root_dir == "final_valid_data":
            # read image and labels
            img_name = os.path.join("final_valid_data", str(idx + 1) + '.npy')
            label_name = os.path.join("final_valid_label", str(idx + 1) + '.npy')

        if self.root_dir == "final_test_data":
            img_name = os.path.join("final_test_data", str(idx + 1) + '.npy')
            label_name = os.path.join("final_test_label", str(idx + 1) + '.npy')

            # img = cv.imread(img_name, cv.IMREAD_COLOR)
            # label = cv.imread(label_name, cv.IMREAD_GRAYSCALE)

        # img = Image.open(img_name)
        # label = Image.open(label_name)

        img = np.load(img_name)
        label = np.load(label_name)

        # if "train_data" in self.root_dir:
        #     angle = random.randint(0, 90)
        #     img = transforms.functional.rotate(img, angle, InterpolationMode.BILINEAR)
        #     label = transforms.functional.rotate(label, angle, InterpolationMode.NEAREST)
            # transform_list = [transforms.CenterCrop((224, 224))]
            # transform = transforms.Compose(transform_list)
            # img = transform(img)
            # label = transform(label)

            # img.save("test_data.jpg")
            # label.save("test_label.jpg")

            # img = img.convert("RGB")

        # img = np.array(img)
        # label = np.array(label)
            


            # if img.shape[0] < 224:
            #     top_size, bottom_size, left_size, right_size = ((224 - img.shape[0]) // 2, (224 - img.shape[0]) // 2, 0, 0)
            #     img = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            #     label = cv.copyMakeBorder(label, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            # if img.shape[1] < 224:
            #     top_size, bottom_size, left_size, right_size = (0, 0, (224 - img.shape[1]) // 2, (224 - img.shape[1]) // 2)
            #     img = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            #     label = cv.copyMakeBorder(label, top_size,bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            
            # label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)

            # img = img[(img.shape[0] // 2) - 75 : (img.shape[0] // 2) + 75, (img.shape[1] // 2) - 75 : (img.shape[1] // 2) + 75]
            # label = label[(label.shape[0] // 2) - 75 : (label.shape[0] // 2) + 75, (label.shape[1] // 2) - 75 : (label.shape[1] // 2) + 75]
            # img = img.transpose(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.view(1, 1, label.shape[0], label.shape[1])
        label1 = torch.where(label==0, torch.tensor(1), torch.tensor(0))
        label2 = torch.where(label==85, torch.tensor(1), torch.tensor(0))
        label3 = torch.where(label==170, torch.tensor(1), torch.tensor(0))
        label4 = torch.where(label==255, torch.tensor(1), torch.tensor(0))
        new_label = torch.cat((label1, label2, label3, label4))
        img = torch.tensor(img, dtype=torch.float32)
        # img = torch.div(img, 255)
        img = img.view(1, 1, img.shape[0], img.shape[1])
        return img, new_label
        


def get_train_data_loader(batch_size, save_path):
    train_data = MMdataset(save_path)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return dataloader


class VAEDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx + 1) + '.npy')
        # img = Image.open(img_name)
        img = np.load(img_name)
        # angle = random.randint(0, 90)
        # img = transforms.functional.rotate(img, angle, InterpolationMode.BILINEAR)
        # img = cv.imread(img_name, cv.IMREAD_COLOR)
        # transform_list = [transforms.CenterCrop((224, 224))]
        # transform = transforms.Compose(transform_list)
        # img = transform(img)
        # img = img.convert("RGB")
        # img = np.array(img)
        # img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)
        # img = torch.div(img, 255)
        img = img.view(1, img.shape[0], img.shape[1])
        return img

def get_generate_data_loader(batch_size, save_path):
    train_data = VAEDataset(save_path)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return dataloader


class MMdataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def __len__(self):
        if "data" in self.root_dir:
            return len(os.listdir(self.root_dir))
        else:
            return len(os.listdir(self.root_dir+"/data"))

    def __getitem__(self, idx):
        img_name = ""
        label_name = ""
        if  "data" in self.root_dir:
            # read image and labels
            img_name = os.path.join(self.root_dir, str(idx + 1) + '.npy')
            label_name = os.path.join(self.root_dir[:16]+"/label", str(idx + 1) + '.npy')
        if self.root_dir == "new_mm_valid":
            # read image and labels
            img_name = os.path.join("new_mm_valid/data", str(idx + 1) + '.npy')
            label_name = os.path.join("new_mm_valid/label", str(idx + 1) + '.npy')

        if self.root_dir == "new_mm_test":
            img_name = os.path.join("new_mm_test/data", str(idx + 1) + '.npy')
            label_name = os.path.join("new_mm_test/label", str(idx + 1) + '.npy')

            # img = cv.imread(img_name, cv.IMREAD_COLOR)
            # label = cv.imread(label_name, cv.IMREAD_GRAYSCALE)

        # img = Image.open(img_name)
        # label = Image.open(label_name)

        img = np.load(img_name)
        label = np.load(label_name)

        # if "train_data" in self.root_dir:
        #     angle = random.randint(0, 90)
        #     img = transforms.functional.rotate(img, angle, InterpolationMode.BILINEAR)
        #     label = transforms.functional.rotate(label, angle, InterpolationMode.NEAREST)
            # transform_list = [transforms.CenterCrop((224, 224))]
            # transform = transforms.Compose(transform_list)
            # img = transform(img)
            # label = transform(label)

            # img.save("test_data.jpg")
            # label.save("test_label.jpg")

            # img = img.convert("RGB")

        # img = np.array(img)
        # label = np.array(label)
            


            # if img.shape[0] < 224:
            #     top_size, bottom_size, left_size, right_size = ((224 - img.shape[0]) // 2, (224 - img.shape[0]) // 2, 0, 0)
            #     img = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            #     label = cv.copyMakeBorder(label, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            # if img.shape[1] < 224:
            #     top_size, bottom_size, left_size, right_size = (0, 0, (224 - img.shape[1]) // 2, (224 - img.shape[1]) // 2)
            #     img = cv.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            #     label = cv.copyMakeBorder(label, top_size,bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=(0, 0, 0))
            
            # label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)

            # img = img[(img.shape[0] // 2) - 75 : (img.shape[0] // 2) + 75, (img.shape[1] // 2) - 75 : (img.shape[1] // 2) + 75]
            # label = label[(label.shape[0] // 2) - 75 : (label.shape[0] // 2) + 75, (label.shape[1] // 2) - 75 : (label.shape[1] // 2) + 75]
            # img = img.transpose(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.view(1, 1, label.shape[0], label.shape[1])
        label1 = torch.where(label==0, torch.tensor(1), torch.tensor(0))
        label2 = torch.where(label==85, torch.tensor(1), torch.tensor(0))
        label3 = torch.where(label==170, torch.tensor(1), torch.tensor(0))
        label4 = torch.where(label==255, torch.tensor(1), torch.tensor(0))
        new_label = torch.cat((label1, label2, label3, label4))
        img = torch.tensor(img, dtype=torch.float32)
        # img = torch.div(img, 255)
        img = img.view(1, 1, img.shape[0], img.shape[1])
        return img, new_label
        


