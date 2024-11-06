import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from model import UNet2D
import torch.nn.functional as F
import copy
from vae import VAE
import utils
import numpy as np
import random

kl_weight = 0.000025

def cross_entropy_loss(pred, target):
    loss_function = nn.CrossEntropyLoss() 
    num = pred.shape[0]
    pred = pred.transpose(1, 4)
    target = target.transpose(1, 4)
    pred = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2]*pred.shape[3], pred.shape[4])
    target = target.reshape(target.shape[0]*target.shape[1]*target.shape[2]*target.shape[3], target.shape[4])
    count = 0
    label = torch.nonzero(target)
    label = label[:, 1]
    loss = loss_function(pred, label)

    return loss

def total_dice_loss(pred, target):
    dice_loss_lv = dice_loss(pred[:, 0, 0, :, :], target[:, 0, 0, :, :])
    dice_loss_myo = dice_loss(pred[:, 1, 0, :, :], target[:, 1, 0, :, :])
    dice_loss_rv = dice_loss(pred[:, 2, 0, :, :], target[:, 2, 0, :, :])
    dice_loss_bg = dice_loss(pred[:, 3, 0, :, :], target[:, 3, 0, :, :])
    loss = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

    return loss

def dice_loss(pred, target):
    dice = 0
    for i in range(pred.shape[0]):
        dice += dice_coeff(pred[i], target[i])

    return 1 - dice / pred.shape[0]

def dice_coeff(pred, target):
    smooth = 1e-10
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((2 * intersection + smooth) / (ifflat.sum() + tfflat.sum() + smooth))

    return score

def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    
    return loss


    
class Agent():
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset = utils.ACDCDataset("labeled_data/train_data_" + str(id))
        self.train_loader = utils.get_train_data_loader(args.batch_size, "labeled_data/train_data_" + str(id))
        self.cos_dataset = utils.VAEDataset("all_data/train_data_" + str(id))
        self.cos_loader = utils.get_generate_data_loader(20, "all_data/train_data_" + str(id))
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
     
        self.n_data = len(self.dataset)
        self.n_generate_data = len(self.cos_dataset)
        self.seg_model = UNet2D(args.in_channels, args.out_channels, final_sigmoid=False)
        self.seg_model.to(self.device)


    def train_local(self, global_model, global_generate_model, lam):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        initial_generate_model_params = parameters_to_vector(global_generate_model.parameters()).detach()
        global_model.train()  
        global_generate_model.train()     

        optimizer1 = torch.optim.Adam(global_model.parameters(), lr=self.args.learning_rate)
        optimizer2 = torch.optim.Adam(global_generate_model.parameters(), lr=0.001)

        loss_function = nn.MSELoss()
      
        unlabel = []
        for j, unlabel_data in enumerate(self.cos_loader):
            unlabel.append(unlabel_data)
    

        for generate_epoch in range(3):
            generateloss = 0
            for l, train_unlabel_data in enumerate(self.cos_loader):
                train_unlabel_data = train_unlabel_data.to(self.device)
                y, mean, logvar, z = global_generate_model(train_unlabel_data)
                generate_loss = loss_fn(train_unlabel_data, y, mean, logvar)
                generateloss += generate_loss * 20
                
                optimizer2.zero_grad()
                generate_loss.backward()
                optimizer2.step()


        for epoch in range(self.args.epoch):
            for i, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                transform_data = data.view(data.shape[0], 1, data.shape[3], data.shape[4])

                y_hat, mean, logvar, z1 = global_generate_model(transform_data)

                outputs = global_model(data, z1)
                loss1 = total_dice_loss(outputs, label)
                loss2 = cross_entropy_loss(outputs, label)
                
                unlabel_data = unlabel[i].to(self.device)

                aug_data, mean, logvar, z2 = global_generate_model(unlabel_data)

                unlabel_data = unlabel_data.view(unlabel_data.shape[0], unlabel_data.shape[1], 1, unlabel_data.shape[2], unlabel_data.shape[3])
                aug_data = aug_data.detach()
                aug_data = aug_data.view(aug_data.shape[0], aug_data.shape[1], 1, aug_data.shape[2], aug_data.shape[3])

                pseudo_label = global_model(unlabel_data, z2)
                pseudo_label = pseudo_label.detach()
           
                output = global_model(aug_data, z2)


                cos_loss = loss_function(output, pseudo_label)

                loss = loss1 + loss2 + lam * 20 * cos_loss


                optimizer1.zero_grad()  
                optimizer2.zero_grad()  
                loss.backward()
                optimizer2.step()
                optimizer1.step()

        with torch.no_grad():
            update1 = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            update2 = parameters_to_vector(global_generate_model.parameters()).double() - initial_generate_model_params
            return update1, update2

    
  