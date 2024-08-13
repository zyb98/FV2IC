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

def kl_loss(pred, target):
    loss_function = nn.KLDivLoss()
    pred = pred.transpose(1, 4)
    target = target.transpose(1, 4)
    pred = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2]*pred.shape[3], pred.shape[4])
    target = target.reshape(target.shape[0]*target.shape[1]*target.shape[2]*target.shape[3], target.shape[4])
    loss = loss_function(pred.log(), target)

    return loss

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

def dice_coeff(pred, target):
    smooth = 1e-10
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((2 * intersection + smooth) / (ifflat.sum() + tfflat.sum() + smooth))
    
    return score

def dice_loss(pred, target):
    dice = 0
    for i in range(pred.shape[0]):
        dice += dice_coeff(pred[i], target[i])
    return 1 - dice / pred.shape[0]


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    # print(recons_loss, kl_loss)
    # print(recons_loss)
    # print(kl_loss)
    loss = recons_loss + kl_loss * kl_weight
    return loss


def sharpening(pred, t):
    new_pre = torch.pow(pred, 1 / t)
    tmp = torch.sum(new_pre, dim=1)
    tmp = tmp.view(tmp.shape[0], tmp.shape[1], 1, tmp.shape[2], tmp.shape[3])
    new_tmp = torch.cat((tmp, tmp, tmp, tmp), 1)
    get_pre = torch.div(new_pre, new_tmp)

    return get_pre

    
class Agent():
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset = utils.ACDCDataset("upper_data_5%/train_data_" + str(id))
        self.train_loader = utils.get_train_data_loader(args.batch_size, "upper_data_5%/train_data_" + str(id))
        # self.generate_dataset = utils.VAEDataset("new_exp_generate_data/generate_data_" + str(id))
        # self.generate_loader = utils.get_generate_data_loader(args.batch_size, "new_exp_generate_data/generate_data_" + str(id))
        self.cos_dataset = utils.VAEDataset("upper_data3/train_data_" + str(id))
        self.cos_loader = utils.get_generate_data_loader(20, "upper_data3/train_data_" + str(id))
        self.device = torch.device("cuda:6" if torch.cuda.is_available else "cpu")
        # self.device = torch.device("cpu")
        self.n_data = len(self.dataset)
        self.n_generate_data = len(self.cos_dataset)
        self.seg_model = UNet2D(args.in_channels, args.out_channels, final_sigmoid=False)
        self.seg_model.to(self.device)
        # self.optimizer1 = torch.optim.Adam(global_model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)
        # self.optimizer2 = torch.optim.Adam(global_model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)
        # self.generate_model = VAE()
        # self.generate_model.load_state_dict(torch.load("global_generate_model.pth"))
        # self.generate_model.to(self.device)

    def train_local(self, global_model, global_generate_model, lam):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        initial_generate_model_params = parameters_to_vector(global_generate_model.parameters()).detach()
        global_model.train()  
        global_generate_model.train()     
        optimizer1 = torch.optim.Adam(global_model.parameters(), lr=self.args.learning_rate)
        optimizer2 = torch.optim.Adam(global_generate_model.parameters(), lr=0.001)
        # loss_function = SoftDiceLoss()
        # loss_function1 = nn.KLDivLoss(reduction = 'batchmean')
        loss_function1 = nn.MSELoss()
        loss_function2 = nn.MSELoss()

        # generate_model = VAE()
        # generate_model.load_state_dict(torch.load("mean_teacher_generate_model.pth"))
        # generate_model = generate_model.to(self.device)

        z_data = []
        # ret_label = []
        # get_Z = []

        # pseudo = []
        unlabel = []
        for j, unlabel_data in enumerate(self.cos_loader):
            unlabel.append(unlabel_data)
            # unlabel_data = unlabel_data.to(self.device)
            # data = unlabel_data.view(20, 1, 1, 224, 224)
            # pseudo_label = global_model(data, z)
            # pseudo_label = pseudo_label.detach()
            # pseudo.append(pseudo_label)
        

        diceloss = 0
        celoss = 0
        generateloss = 0
        cosloss = 0
        reconsegloss = 0

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
            # print(epoch)
            # for i in range(self.args.num_agents):
            #     if i == self.id:
            #         continue
            #     else:
            #         inter_label += generate_output_dict[i]
            # inter_label = torch.div(inter_label, 9)
            # inter_label = sharpening(inter_label, 0.1)
            # inter_label = inter_label.detach()
            
            # print(label)
            for i, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                transform_data = data.view(data.shape[0], 1, data.shape[3], data.shape[4])

                y_hat, mean, logvar, z1 = global_generate_model(transform_data)

                # z1 = z1.detach()
                # label_generate_loss = loss_fn(transform_data, y_hat, mean, logvar)

                # y_hat = y_hat.view(y_hat.shape[0], 1, y_hat.shape[1], y_hat.shape[2], y_hat.shape[3])
                # recons_seg_output = torch.utils.checkpoint.checkpoint(global_model, y_hat, z1)
                # recons_seg_loss = total_dice_loss(recons_seg_output, label)
                # reconsegloss += 4 * recons_seg_loss


                outputs = global_model(data, z1)
                # outputs = global_model(data)
                loss1 = total_dice_loss(outputs, label)
                loss2 = cross_entropy_loss(outputs, label)

                diceloss += 4 * loss1
                celoss += 4 * loss2

                
                unlabel_data = unlabel[i].to(self.device)

                # recons_data = recons[i]
                # recons_data = recons_data.view(recons_data.shape[0], 1, 1, recons_data.shape[2], recons_data.shape[3])
                # # new_recons = torch.div(recons_data, 255)
                # # new_recons = new_recons.detach()
                # get_z_data = get_z[i].to(self.device)
                # # get_z_data = get_z_data.detach()

                # # inter_pesudo_label = global_model(new_recons, get_z_da)
                # # inter_pesudo_label = sharpening(inter_pesudo_label, 0.1)
                # # inter_pesudo_label = inter_pesudo_label.detach()

                # inter_pesudo_label = get_label[i].to(self.device)

                # # inter_pesudo_label = sharpening(inter_pesudo_label, 0.1)
                # inter_pesudo_label = inter_pesudo_label.detach()

                # # aug_recons = recons_data + torch.normal(0, 25, (16, 1, 1, 224, 224)).to(self.device)
                # # aug_recons = torch.clip(aug_recons, 0, 255)
                # # aug_recons = aug_recons.detach()
                # # new_aug_recons = torch.div(aug_recons, 255)

                # out_put = global_model(recons_data, get_z_data)
                
                # inter_cos_loss = loss_function2(out_put, inter_pesudo_label)




                # print(unlabel_data)
                # new_unlabel_data = unlabel_data * 255

                aug_data, mean, logvar, z2 = global_generate_model(unlabel_data)
                # _, mean, logvar, z2 = global_generate_model(unlabel_data)
                # aug_data, _, _, _ = generate_model(unlabel_data)

                # new_z = z2.detach()
                # aug_data = global_generate_model.get_recons(new_z)

                # z_data.append(z2.detach())

                # generate_loss = loss_fn(unlabel_data, aug_data, mean, logvar)
                # generateloss += generate_loss * 20
                unlabel_data = unlabel_data.view(unlabel_data.shape[0], unlabel_data.shape[1], 1, unlabel_data.shape[2], unlabel_data.shape[3])
                # gauss_noise = torch.randn(unlabel_data.shape).to(self.device)
                # gauss_noise = torch.normal(0, 0.1, (20, 1, 1, 224, 224)).to(self.device)
                # aug_data = unlabel_data + gauss_noise
                # aug_data = torch.clamp(aug_data, 0, 1)
                aug_data = aug_data.detach()
                aug_data = aug_data.view(aug_data.shape[0], aug_data.shape[1], 1, aug_data.shape[2], aug_data.shape[3])

                # new_unlabel_data = torch.div(unlabel_data, 255)
                # aug_data = torch.div(aug_data, 255)
                # z_data.append(z2)

                # pseudo_label = global_model(unlabel_data, z2)
                pseudo_label = global_model(unlabel_data, z2)
                pseudo_label = pseudo_label.detach()
                # pseudo_label = pseudo[i]

                # print(pseudo_label[0][0][0][0][0], pseudo_label[0][1][0][0][0], pseudo_label[0][2][0][0][0], pseudo_label[0][3][0][0][0])
                

                # gauss = torch.normal(0, 5, (20, 1, 1, 224, 224)).to(self.device)

                # gauss_aug_data = unlabel_data + gauss
                # gauss_aug_data = torch.clip(gauss_aug_data, 0, 255)
                # gauss_aug_data = torch.div(gauss_aug_data, 255)
                # pseudo_label = global_model(gauss_aug_data)
                # pseudo_label = pseudo_label.detach()
                # ret_label.append(pseudo_label)
                # pseudo_label = sharpening(pseudo_label, 0.1)



                # print(pseudo_label[0][0][0][0][0], pseudo_label[0][1][0][0][0], pseudo_label[0][2][0][0][0], pseudo_label[0][3][0][0][0])
                # print(" ")
                # output = global_model(aug_data, z2)
                output = global_model(aug_data, z2)

                # getz = get_z[i].to(self.device)

                cos_loss = loss_function1(output, pseudo_label)
                cosloss += cos_loss * 20
                # getZ = 0.5 * getz + 0.5 * output
                # getZ = getZ.detach()
                # get_Z.append(getZ)
                # cos_loss = cos_loss / 50176
                # inter_output = global_model(train_generate_data, z)
                # inter_cos_loss = loss_function2(inter_output, inter_label)

                # print(loss1.item(), loss2.item(), cos_loss.item())

                # loss = 0
                # if self.id == 0 or self.id == 1:
                #     loss = loss1 + loss2 + 100 * cos_loss
                # else:
                #     loss = 1000 * cos_loss

                loss = loss1 + loss2 + lam * 20 * cos_loss
                # print(cos_loss)
                # loss = loss1 + loss2
                # generate_loss = 0.2 * generate_loss1 + 0.8 * generate_loss2



                optimizer1.zero_grad()  
                optimizer2.zero_grad()  
                loss.backward()
                optimizer2.step()
                optimizer1.step()


            
        # with torch.no_grad():
        #     update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
        #     return update
        # vector_to_parameters(copy.deepcopy(parameters_to_vector(global_model.parameters()).detach()), self.seg_model.parameters())
        # with torch.no_grad():
        #     update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
        #     return update

        diceloss = diceloss / 20
        celoss = celoss / 20
        generateloss = generateloss / 100
        cosloss = cosloss / 5

        with torch.no_grad():
            update1 = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            update2 = parameters_to_vector(global_generate_model.parameters()).double() - initial_generate_model_params
            return update1, update2, diceloss, celoss, generateloss, cosloss
            # return update1
    
    def train_generate_model(self, global_generate_model):
        initial_generate_model_params = parameters_to_vector(global_generate_model.parameters()).detach()
        global_generate_model.train()
        optimizer = torch.optim.Adam(global_generate_model.parameters(), lr=0.001)

        for epoch in range(6):
            for i, data in enumerate(self.generate_loader):
                data = data.to(self.device)
                y_hat, mean, logvar, _ = global_generate_model(data)
                loss = loss_fn(data, y_hat, mean, logvar)

                optimizer.zero_grad()  
                loss.backward() 
                optimizer.step()  
        
        with torch.no_grad():
            update = parameters_to_vector(global_generate_model.parameters()).double() - initial_generate_model_params
            return update

    def train_generate_data(self, generate_output_dict, global_model, get_generate_data, lam):
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        initial_seg_model_params = parameters_to_vector(self.seg_model.parameters()).detach()
        self.seg_model.train()
        optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=self.args.learning_rate)
        label = torch.zeros(generate_output_dict[0].shape)
        label = label.to(device)
        # generate_output_dict = np.array(generate_output_dict)
        # generate_output_dict = torch.tensor(generate_output_dict, dtype=torch.float32)
        # print(generate_output_dict.shape)
        for i in range(self.args.num_agents):
            if i == self.id:
                continue
            else:
                label += generate_output_dict[i]
        label = label.detach()
        label = torch.div(label, 9)
        label = sharpening(label, 0.1)
        # print(label)
        outputs = self.seg_model(get_generate_data)
        loss_function = nn.MSELoss()
        loss = 10 * loss_function(outputs, label)

        # print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            update1 = initial_seg_model_params - initial_global_model_params
            update2 = parameters_to_vector(self.seg_model.parameters()).double() - initial_seg_model_params
            # zero = torch.nonzero(update2)
            update = update1 + lam * update2
            return update

    
    def intra_client(self, update_dict, global_model):
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=self.args.learning_rate)
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        initial_seg_model_params = parameters_to_vector(self.seg_model.parameters()).detach()
        self.seg_model.train()
        cur_model = UNet2D(self.args.in_channels, self.args.out_channels, final_sigmoid=False).to(device)
        for i in range(self.args.num_agents):
            for j, unlabel_data in enumerate(self.cos_loader):
                unlabel_data = unlabel_data.to(device)
                unlabel_data = unlabel_data.view(unlabel_data.shape[0], unlabel_data.shape[1], 1, unlabel_data.shape[2], unlabel_data.shape[3])
                output = self.seg_model(unlabel_data)
                pseudo_label = torch.zeros(output.shape).to(device)
                for z in range(10):
                    if z == self.id:
                        continue
                    else:
                        vector_to_parameters((initial_global_model_params.double() - update_dict[z]).float(), cur_model.parameters())
                        get_pseudo_label = cur_model(unlabel_data)
                        pseudo_label += get_pseudo_label
                    
                pseudo_label = torch.div(pseudo_label, 9)
                pseudo_label = sharpening(pseudo_label, 0.1)
                pseudo_label = pseudo_label.detach()
                loss_function = nn.MSELoss()
                loss = 10 * loss_function(output, pseudo_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            update1 = initial_seg_model_params - initial_global_model_params
            update2 = parameters_to_vector(self.seg_model.parameters()).double() - initial_seg_model_params
            # zero = torch.nonzero(update2)
            update = update1 + update2
            return update

                
            
