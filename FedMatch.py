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
from model import UNet2D
from tqdm import tqdm
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from sklearn import metrics
from setting import parse_opt
import cv2 as cv
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from vae import VAE
from torchvision.transforms import ToPILImage
from utils import SoftDiceLoss
from torch.autograd import Function
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from PIL import Image
import utils
import os
import random
import pandas as pd
from medpy.metric.binary import dc, hd95, assd


def seed_torch(seed=666):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
 
seed_torch()

kl_weight = 0.000025

sets = parse_opt()

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


class Aggregation():
    def __init__(self, agent_size, n_params, args):
        self.agent_data_sizes = agent_size
        self.n_params = n_params
        self.args = args
        self.server_lr = 1.0
    
    def aggregate_update(self, global_model, agent_updates_dict):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(torch.device("cuda:7"))
        model_bank = []

        aggregated_updates = 0
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)



        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())

        for i in range(10):
            tmp_model_params = (cur_global_params + agent_updates_dict[i])
            model_bank.append(tmp_model_params.float())

        return model_bank

    

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    
class Agent():
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.dataset = utils.ACDCDataset("upper_data_5%/train_data_" + str(id))
        self.train_loader = utils.get_train_data_loader(args.batch_size, "upper_data_5%/train_data_" + str(id))
        self.cos_dataset = utils.VAEDataset("upper_data3/train_data_" + str(id))
        self.cos_loader = utils.get_generate_data_loader(20, "upper_data3/train_data_" + str(id))
        self.device = torch.device("cuda:7" if torch.cuda.is_available else "cpu")
        self.n_data = len(self.dataset)
        self.n_generate_data = len(self.cos_dataset)
        self.seg_model = UNet2D(args.in_channels, args.out_channels, final_sigmoid=False)
        self.seg_model.to(self.device)
        self.model_bank = []

    def train_local(self, global_model1, global_model2, agent_updates_dict, lam):

        tmp_model = UNet2D(1, 4, final_sigmoid=False).to(torch.device("cuda:7"))
        initial_global_model_params1 = parameters_to_vector(global_model1.parameters()).detach()
        initial_global_model_params2 = parameters_to_vector(global_model2.parameters()).detach()
        global_model1.train()
        global_model2.train()
        generate_model = VAE()
        generate_model.load_state_dict(torch.load("mean_teacher_generate_model.pth"))
        generate_model = generate_model.to(self.device)

        optimizer1 = torch.optim.Adam(global_model1.parameters(), lr=self.args.learning_rate)
        optimizer2 = torch.optim.Adam(global_model2.parameters(), lr=self.args.learning_rate)
        loss_function = nn.MSELoss()

        unlabel = []
        for j, unlabel_data in enumerate(self.cos_loader):
            unlabel.append(unlabel_data)
        
        self.model_bank = agent_updates_dict


        for epoch in range(self.args.epoch):
            for i, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                
                outputs = global_model1(data)
                loss1 = total_dice_loss(outputs, label)
                loss2 = cross_entropy_loss(outputs, label)


                
                unlabel_data = unlabel[i].to(self.device)
                aug_data, _, _, _ = generate_model(unlabel_data)
                unlabel_data  = unlabel_data.view(20, 1, 1, 224, 224)
                # aug_data = unlabel_data + torch.normal(0, 20, (20, 1, 1, 224, 224)).to(torch.device("cuda:0"))
                # aug_data = torch.clip(aug_data, 0, 255)
                aug_data = aug_data.detach()
                aug_data = aug_data.view(20, 1, 1, 224, 224)

                # res = torch.zeros(20, 4, 1, 224, 224).to(torch.device("cuda:0"))
                # for j in self.model_bank:
                #     vector_to_parameters(j, tmp_model.parameters())
                #     output = tmp_model(unlabel_data)
                #     output = output.detach()
                #     res += output
                
                # res = torch.div(res, 10)
                # res = res.detach()
                res = global_model2(unlabel_data)


                out_put = global_model2(aug_data)
                loss3 = loss_function(out_put, res)

                model1_loss = loss1 + loss2 
                model2_loss = 20 * lam * loss3 

                optimizer1.zero_grad()
                model1_loss.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                model2_loss.backward()
                optimizer2.step()

            

        with torch.no_grad():
            update1 = parameters_to_vector(global_model1.parameters()).double() - initial_global_model_params1
            update2 = parameters_to_vector(global_model2.parameters()).double() - initial_global_model_params2
            return update1, update2



def train():
    df = pd.DataFrame(columns=['round', 'loss', 'dice', 'dice1', 'dice2', 'dice3'])
    df.to_csv("FedMatch_results.csv", index=False)
    sets = parse_opt()
    # train_data = utils.get_data_loader(sets.batch_size, 'train_data')
    test_data = utils.get_train_data_loader(sets.batch_size, 'final_valid_data')
    device = torch.device("cuda:7" if torch.cuda.is_available else "cpu")
    # device = torch.device('cpu')
    global_model1 = UNet2D(sets.in_channels, sets.out_channels, final_sigmoid=False)
    global_model1 = global_model1.to(device)
    global_model2 = UNet2D(sets.in_channels, sets.out_channels, final_sigmoid=False).to(device)
    agents, agent_data_size, agent_generate_data_size = [], {}, {}
    for agent_id in range(sets.num_agents):
        agent = Agent(agent_id, sets)
        agent_data_size[agent_id] = agent.n_data
        agents.append(agent)
    n_model_params = len(parameters_to_vector(global_model1.parameters()))
    aggregator = Aggregation(agent_data_size, n_model_params, sets)
    best_dice = 0.0
    model_bank_dict = []


    for rnd in range(1, sets.rounds + 1):
        # lam = 0.005 * rnd
        lam = 1 / (np.exp(40 - rnd) + 1)
        print(rnd, lam)
        epoch_start = time.time()
        rnd_global_params1 = parameters_to_vector(global_model1.parameters()).detach()
        rnd_global_params2 = parameters_to_vector(global_model2.parameters()).detach()
        agent_updates_dict1 = {}
        agent_updates_dict2 = {}


        for agent_id in tqdm(range(sets.num_agents)):
            # print(agent_id)
            update1, update2 = agents[agent_id].train_local(global_model1, global_model2, model_bank_dict, lam)

            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params1), global_model1.parameters())
            vector_to_parameters(copy.deepcopy(rnd_global_params2), global_model2.parameters())

            agent_updates_dict1[agent_id] = update1
            agent_updates_dict2[agent_id] = update2

        
        model_bank_dict = aggregator.aggregate_update(global_model1, agent_updates_dict1)
        aggregator.aggregate_update(global_model2, agent_updates_dict2)

        

    

        valid_loss = 0.0
        valid_dice = 0.0
        dice1 = 0.0
        dice2 = 0.0
        dice3 = 0.0

        with torch.no_grad():
            global_model1.eval()
            global_model2.eval()

            # final_model = UNet2D(1, 4, final_sigmoid=False).to(device)
            # param1 = parameters_to_vector(global_model1.parameters()).detach()
            # param2 = parameters_to_vector(global_model2.parameters()).detach()
            # param = param1.float() + param2.float()
            # vector_to_parameters(param, final_model.parameters())

            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs1 = global_model1(inputs)
                outputs2 = global_model2(inputs)
                out_put = outputs1 + outputs2
                # outputs = global_model(inputs)
                loss = total_dice_loss(out_put, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, pre = torch.max(out_put, 1)
                res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res = torch.cat((res1, res2, res3, res4), 1)
                res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])
                # res = torch.where(outputs>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
                # valid_dice += Dice_Coeff(res, labels) * inputs.size(0)
                dice1 += Dice_Coeff1(res, labels) * inputs.size(0)
                dice2 += Dice_Coeff2(res, labels) * inputs.size(0)
                dice3 += Dice_Coeff3(res, labels) * inputs.size(0)
        
        avg_valid_loss = valid_loss / sets.test_size
        # avg_valid_dice = valid_dice / sets.test_size
        avg_valid_dice1 = dice1 / sets.test_size
        avg_valid_dice2 = dice2 / sets.test_size
        avg_valid_dice3 = dice3 / sets.test_size
        avg_valid_dice = (avg_valid_dice1 + avg_valid_dice2 + avg_valid_dice3) / 3


        if best_dice < avg_valid_dice:
            best_dice = avg_valid_dice
            torch.save(global_model1.state_dict(), "FedMatch_global_model1_5%.pth")
            torch.save(global_model2.state_dict(), "FedMatch_global_model2_5%.pth")
        


        epoch_end = time.time()
        print("Round: {:03d}, Validation: Loss: {:.4f} Dice: {:4f}, Dice1: {:.4f}, Dice2: {:.4f}, Dice3: {:.4f} Time: {:.4f}s".format(
            rnd, avg_valid_loss, avg_valid_dice, avg_valid_dice1, avg_valid_dice2, avg_valid_dice3, epoch_end-epoch_start
        ))
        print("Best Dice:", best_dice.item())
        ndf = pd.DataFrame(columns=["round", "loss", "dice", "dice1", "dice2", "dice3"])
        ndf.loc[0] = [rnd] + [np.array(avg_valid_loss)] + [np.array(avg_valid_dice.cpu())] + [np.array(avg_valid_dice1.cpu())] + [np.array(avg_valid_dice2.cpu())] + [np.array(avg_valid_dice3.cpu())]
        ndf.to_csv("FedMatch_results.csv", mode="a", index=False, header=False)



def total_dice_loss(pred, target):
    dice_loss_lv = dice_loss(pred[:, 0, 0, :, :], target[:, 0, 0, :, :])
    dice_loss_myo = dice_loss(pred[:, 1, 0, :, :], target[:, 1, 0, :, :])
    dice_loss_rv = dice_loss(pred[:, 2, 0, :, :], target[:, 2, 0, :, :])
    dice_loss_bg = dice_loss(pred[:, 3, 0, :, :], target[:, 3, 0, :, :])
    loss = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg

    return loss

def dice_coeff(pred, target):
    smooth = 0.1
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

def DiceCoeff(pred, target):
    smooth = 0.0001
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((2 * intersection + smooth) / (ifflat.sum() + tfflat.sum() + smooth))
    
    return score
    
def Dice_Coeff(input, target):
    """Dice coeff for batches"""
    dice = 0
    for i in range(input.shape[0]):
        dice += DiceCoeff(input[i][1:4], target[i][1:4])
    return dice / input.shape[0]

def Dice_Coeff1(input, target):
    """Dice coeff for batches"""
    dice = 0
    for i in range(input.shape[0]):
        dice += DiceCoeff(input[i][1], target[i][1])
    return dice / input.shape[0]

def Dice_Coeff2(input, target):
    """Dice coeff for batches"""
    dice = 0
    for i in range(input.shape[0]):
        dice += DiceCoeff(input[i][2], target[i][2])
    return dice / input.shape[0]

def Dice_Coeff3(input, target):
    """Dice coeff for batches"""
    dice = 0
    for i in range(input.shape[0]):
        dice += DiceCoeff(input[i][3], target[i][3])
    return dice / input.shape[0]


def JaccardCoeff(pred, target):
    smooth = 0.0001
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((intersection + smooth) / (ifflat.sum() + tfflat.sum() - intersection + smooth))

    return score

def Jaccard_Coeff1(input, target):
    jaccard = 0
    for i in range(input.shape[0]):
        jaccard += JaccardCoeff(input[i][1], target[i][1])
    return jaccard / input.shape[0]

def Jaccard_Coeff2(input, target):
    jaccard = 0
    for i in range(input.shape[0]):
        jaccard += JaccardCoeff(input[i][2], target[i][2])
    return jaccard / input.shape[0]

def Jaccard_Coeff3(input, target):
    jaccard = 0
    for i in range(input.shape[0]):
        jaccard += JaccardCoeff(input[i][3], target[i][3])
    return jaccard / input.shape[0]


def Accuracy(pred, target):
    accuracy = 0
    for i in range(pred.shape[0]):
        intersection = (pred[i].contiguous().view(-1) * target[i].contiguous().view(-1)).sum()
        accuracy += intersection / 50176
    
    return accuracy / pred.shape[0]

def SensitiveCoeff(pred, target):
    smooth = 0.0001
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((intersection + smooth) / (tfflat.sum() + smooth))

    return score

def Sensitive_Coeff1(input, target):
    sensitive = 0
    for i in range(input.shape[0]):
        sensitive += SensitiveCoeff(input[i][1], target[i][1])
    
    return sensitive / input.shape[0]

def Sensitive_Coeff2(input, target):
    sensitive = 0
    for i in range(input.shape[0]):
        sensitive += SensitiveCoeff(input[i][2], target[i][2])
    
    return sensitive / input.shape[0]

def Sensitive_Coeff3(input, target):
    sensitive = 0
    for i in range(input.shape[0]):
        sensitive += SensitiveCoeff(input[i][3], target[i][3])
    
    return sensitive / input.shape[0]


# def test():
#     device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#     global_model1 = UNet2D(1, 4, final_sigmoid=False)
#     global_model2 = UNet2D(1, 4, final_sigmoid=False)
#     global_model1.load_state_dict(torch.load("FedMatch_global_model1_5%.pth"))
#     global_model2.load_state_dict(torch.load("FedMatch_global_model2_5%.pth"))
#     global_model1 = global_model1.to(device)
#     global_model2 = global_model2.to(device)
#     test_data = utils.get_train_data_loader(4, 'final_test_data')

#     valid_loss = 0.0
#     valid_dice = 0.0
#     dice1 = 0.0
#     dice2 = 0.0
#     dice3 = 0.0
#     jaccard1 = 0.0
#     jaccard2 = 0.0
#     jaccard3 = 0.0
#     sensitive1 = 0.0
#     sensitive2 = 0.0
#     sensitive3 = 0.0
#     accuracy = 0.0

#     with torch.no_grad():
#             global_model1.eval()
#             global_model2.eval()

#             for j, (inputs, labels) in enumerate(test_data):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 outputs1 = global_model1(inputs)
#                 outputs2 = global_model2(inputs)
#                 out_put = outputs1 + outputs2

#                 # outputs = global_model(inputs)
                
#                 loss = total_dice_loss(out_put, labels)
#                 valid_loss += loss.item() * inputs.size(0)
#                 _, pre = torch.max(out_put, 1)
#                 res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
#                 res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
#                 res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
#                 res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
#                 res = torch.cat((res1, res2, res3, res4), 1)
#                 res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])
#                 # res = torch.where(outputs>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
#                 # valid_dice += Dice_Coeff(res, labels) * inputs.size(0)
#                 dice1 += Dice_Coeff1(res, labels) * inputs.size(0)
#                 dice2 += Dice_Coeff2(res, labels) * inputs.size(0)
#                 dice3 += Dice_Coeff3(res, labels) * inputs.size(0)

#                 jaccard1 += Jaccard_Coeff1(res, labels) * inputs.size(0)
#                 jaccard2 += Jaccard_Coeff2(res, labels) * inputs.size(0)
#                 jaccard3 += Jaccard_Coeff3(res, labels) * inputs.size(0)

#                 sensitive1 += Sensitive_Coeff1(res, labels) * inputs.size(0)
#                 sensitive2 += Sensitive_Coeff2(res, labels) * inputs.size(0)
#                 sensitive3 += Sensitive_Coeff3(res, labels) * inputs.size(0)

#                 accuracy += Accuracy(res, labels) * inputs.size(0)



            
#     avg_valid_loss = valid_loss / 281
#     # avg_valid_dice = valid_dice / 322
#     avg_valid_dice1 = dice1 / 281
#     avg_valid_dice2 = dice2 / 281
#     avg_valid_dice3 = dice3 / 281
#     avg_valid_dice = (avg_valid_dice1 + avg_valid_dice2 + avg_valid_dice3) / 3

#     avg_valid_jaccard1 = jaccard1 / 281
#     avg_valid_jaccard2 = jaccard2 / 281
#     avg_valid_jaccard3 = jaccard3 / 281
#     avg_valid_jaccard = (avg_valid_jaccard1 + avg_valid_jaccard2 + avg_valid_jaccard3) / 3

#     avg_valid_sensitive1 = sensitive1 / 281
#     avg_valid_sensitive2 = sensitive2 / 281
#     avg_valid_sensitive3 = sensitive3 / 281
#     avg_valid_sensitive = (avg_valid_sensitive1 + avg_valid_sensitive2 + avg_valid_sensitive3) / 3

#     avg_valid_accuracy = accuracy / 281
    
#     print("Test: Loss: {:.4f} Dice: {:.4f}, Jaccard: {:.4f}, Sensitive: {:.4f}, Accuracy: {:.4f}".format(
#             avg_valid_loss, avg_valid_dice, avg_valid_jaccard, avg_valid_sensitive, avg_valid_accuracy))

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    global_model1 = UNet2D(1, 4, final_sigmoid=False)
    global_model2 = UNet2D(1, 4, final_sigmoid=False)
    global_model1.load_state_dict(torch.load("FedMatch_global_model1_5%.pth"))
    global_model2.load_state_dict(torch.load("FedMatch_global_model2_5%.pth"))
    global_model1 = global_model1.to(device)
    global_model2 = global_model2.to(device)
    test_data = utils.get_train_data_loader(4, 'final_test_data')

    valid_loss = 0.0
    valid_dice = 0.0
    dice1 = 0.0
    dice2 = 0.0
    dice3 = 0.0
    jaccard1 = 0.0
    jaccard2 = 0.0
    jaccard3 = 0.0
    sensitive1 = 0.0
    sensitive2 = 0.0
    sensitive3 = 0.0
    accuracy = 0.0

    all_dice = 0
    all_hd = 0

    with torch.no_grad():
        global_model1.eval()
        global_model2.eval()
       
        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs1 = global_model1(inputs)
            outputs2 = global_model2(inputs)
            outputs = outputs1 + outputs2
            # outputs = global_model(inputs)

                
            loss = total_dice_loss(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            _, pre = torch.max(outputs, 1)
            new_pre = pre.view(pre.shape[0], pre.shape[2], pre.shape[3])
            _, label = torch.max(labels, 1)
            label = label.view(label.shape[0], label.shape[2], label.shape[3])
            hd = hd95(np.array(new_pre.cpu()), np.array(label.cpu()))
            res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res = torch.cat((res1, res2, res3, res4), 1)
            res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])
            # res = torch.where(outputs>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            # valid_dice += Dice_Coeff(res, labels) * inputs.size(0)
            dice1 += Dice_Coeff1(res, labels) * inputs.size(0)
            dice2 += Dice_Coeff2(res, labels) * inputs.size(0)
            dice3 += Dice_Coeff3(res, labels) * inputs.size(0)

            all_hd += hd * inputs.size(0)

            jaccard1 += Jaccard_Coeff1(res, labels) * inputs.size(0)
            jaccard2 += Jaccard_Coeff2(res, labels) * inputs.size(0)
            jaccard3 += Jaccard_Coeff3(res, labels) * inputs.size(0)

            sensitive1 += Sensitive_Coeff1(res, labels) * inputs.size(0)
            sensitive2 += Sensitive_Coeff2(res, labels) * inputs.size(0)
            sensitive3 += Sensitive_Coeff3(res, labels) * inputs.size(0)

            accuracy += Accuracy(res, labels) * inputs.size(0)





            
    avg_valid_loss = valid_loss / sets.test_size
    # avg_valid_dice = valid_dice / 322
    avg_valid_dice1 = dice1 / sets.test_size
    avg_valid_dice2 = dice2 / sets.test_size
    avg_valid_dice3 = dice3 / sets.test_size
    avg_valid_dice = (avg_valid_dice1 + avg_valid_dice2 + avg_valid_dice3) / 3

    avg_valid_jaccard1 = jaccard1 / sets.test_size
    avg_valid_jaccard2 = jaccard2 / sets.test_size
    avg_valid_jaccard3 = jaccard3 / sets.test_size
    avg_valid_jaccard = (avg_valid_jaccard1 + avg_valid_jaccard2 + avg_valid_jaccard3) / 3

    avg_valid_sensitive1 = sensitive1 / sets.test_size
    avg_valid_sensitive2 = sensitive2 / sets.test_size
    avg_valid_sensitive3 = sensitive3 / sets.test_size
    avg_valid_sensitive = (avg_valid_sensitive1 + avg_valid_sensitive2 + avg_valid_sensitive3) / 3

    avg_valid_accuracy = accuracy / sets.test_size

    avg_hd = all_hd / sets.test_size

    
    print("Test: Loss: {:.4f} Dice: {:.4f}, Jaccard: {:.4f}, Sensitive: {:.4f}, Accuracy: {:.4f}, HD95: {:.4f}".format(
        avg_valid_loss, avg_valid_dice, avg_valid_jaccard, avg_valid_sensitive, avg_valid_accuracy, avg_hd))

if __name__ == '__main__':
    # train()
    test()


   
                
            
