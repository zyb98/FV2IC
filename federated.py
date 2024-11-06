import random
import time
import torch
import copy
import utils
import os
import numpy as np
from vae import VAE
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from agent import Agent
from model import UNet2D
from sklearn import metrics
from setting import parse_opt
from aggregation import Aggregation
from medpy.metric.binary import dc, hd95


def seed_torch(seed=666):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
seed_torch()

sets = parse_opt()

def train():
    valid_data = utils.get_train_data_loader(sets.batch_size, 'valid_data')
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    global_model = UNet2D(sets.in_channels, sets.out_channels, final_sigmoid=False)
    global_model = global_model.to(device)
    global_generate_model = VAE()
    global_generate_model = global_generate_model.to(device)

    agents, agent_data_size, agent_generate_data_size = [], {}, {}
    for agent_id in range(sets.num_agents):
        agent = Agent(agent_id, sets)
        agent_data_size[agent_id] = agent.n_data
        agent_generate_data_size[agent_id] = agent.n_generate_data
        agents.append(agent)
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    n_generate_model_params = len(parameters_to_vector(global_generate_model.parameters()))
    aggregator = Aggregation(agent_data_size, n_model_params, n_generate_model_params)
    
    best_dice = 0.0

    for rnd in range(1, sets.rounds + 1):
        lam = 1 / (np.exp(40 - rnd) + 1)
        print(rnd, lam)
        epoch_start = time.time()
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        rnd_generate_global_params = parameters_to_vector(global_generate_model.parameters()).detach()
        agent_updates_dict = {}
        agent_generate_updates_dict = {}

        for agent_id in tqdm(range(sets.num_agents)):
            update1, update2 = agents[agent_id].train_local(global_model, global_generate_model, lam)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
            vector_to_parameters(copy.deepcopy(rnd_generate_global_params), global_generate_model.parameters())

            agent_updates_dict[agent_id] = update1
            agent_generate_updates_dict[agent_id] = update2

        aggregator.aggregate_update(global_model, global_generate_model, agent_updates_dict, agent_generate_updates_dict, lam)


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


        with torch.no_grad():
            global_model.eval()
            global_generate_model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                new_inputs = inputs.view(inputs.shape[0], 1, inputs.shape[3], inputs.shape[4])
                _, _, _, z = global_generate_model(new_inputs)
                outputs = global_model(inputs, z)
  
                loss = total_dice_loss(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, pre = torch.max(outputs, 1)
                res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res = torch.cat((res1, res2, res3, res4), 1)
                res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])
               
                dice1 += Dice_Coeff(res, labels, 1) * inputs.size(0)
                dice2 += Dice_Coeff(res, labels, 2) * inputs.size(0)
                dice3 += Dice_Coeff(res, labels, 3) * inputs.size(0)
        
        avg_valid_loss = valid_loss / sets.valid_size

        avg_valid_dice1 = dice1 / sets.valid_size
        avg_valid_dice2 = dice2 / sets.valid_size
        avg_valid_dice3 = dice3 / sets.valid_size
        avg_valid_dice = (avg_valid_dice1 + avg_valid_dice2 + avg_valid_dice3) / 3

        if best_dice < avg_valid_dice:
            best_dice = avg_valid_dice
            torch.save(global_model.state_dict(), "ours_global_model.pth")
            torch.save(global_generate_model.state_dict(), "ours_global_generate_model.pth")

        epoch_end = time.time()
        print("Round: {:03d}, Validation: Loss: {:.4f} Dice: {:4f}, Dice1: {:.4f}, Dice2: {:.4f}, Dice3: {:.4f} Time: {:.4f}s".format(
            rnd, avg_valid_loss, avg_valid_dice, avg_valid_dice1, avg_valid_dice2, avg_valid_dice3, epoch_end-epoch_start
        ))
        print("Best Dice:", best_dice.item())


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

def Dice_Coeff(input, target, n):
    """Dice coeff for batches"""
    dice = 0
    for i in range(input.shape[0]):
        dice += DiceCoeff(input[i][n], target[i][n])
    return dice / input.shape[0]

def JaccardCoeff(pred, target):
    smooth = 0.0001
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((intersection + smooth) / (ifflat.sum() + tfflat.sum() - intersection + smooth))

    return score

def Jaccard_Coeff(input, target, n):
    jaccard = 0
    for i in range(input.shape[0]):
        jaccard += JaccardCoeff(input[i][n], target[i][n])
    return jaccard / input.shape[0]


def SensitiveCoeff(pred, target):
    smooth = 0.0001
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((intersection + smooth) / (tfflat.sum() + smooth))

    return score


def Sensitive_Coeff(input, target, n):
    sensitive = 0
    for i in range(input.shape[0]):
        sensitive += SensitiveCoeff(input[i][n], target[i][n])
    
    return sensitive / input.shape[0]


def calculate_average_hd95_batch(pred_batch, gt_batch):
    batch_size = pred_batch.shape[0]
    hd95_values = []

    for i in range(batch_size):
        pred = pred_batch[i]
        gt = gt_batch[i]
        hd95_value = hd95(pred, gt)
        hd95_values.append(hd95_value)
    
    average_hd95 = np.mean(hd95_values)
    return average_hd95


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    global_model = UNet2D(1, 4, final_sigmoid=False)
    global_generate_model = VAE()
    global_model.load_state_dict(torch.load("ours_global_model_20%.pth"))
    global_generate_model.load_state_dict(torch.load("ours_global_generate_model_20%.pth"))
    global_model = global_model.to(device)
    global_generate_model = global_generate_model.to(device)
    test_data = utils.get_train_data_loader(4, 'test_data')

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
        global_model.eval()
        global_generate_model.eval()

        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            new_inputs = inputs.view(inputs.shape[0], 1, inputs.shape[3], inputs.shape[4])
            recons_data, _, _, z = global_generate_model(new_inputs)
            outputs = global_model(inputs, z)
                
            loss = total_dice_loss(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            _, pre = torch.max(outputs, 1)
            new_pre = pre.view(pre.shape[0], pre.shape[2], pre.shape[3])
            _, label = torch.max(labels, 1)
            label = label.view(label.shape[0], label.shape[2], label.shape[3])
            hd = calculate_average_hd95_batch(np.array(new_pre.cpu()), np.array(label.cpu()))
            res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
            res = torch.cat((res1, res2, res3, res4), 1)
            res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])

            dice1 += Dice_Coeff(res, labels, 1) * inputs.size(0)
            dice2 += Dice_Coeff(res, labels, 2) * inputs.size(0)
            dice3 += Dice_Coeff(res, labels, 3) * inputs.size(0)

            all_hd += hd * inputs.size(0)

            jaccard1 += Jaccard_Coeff(res, labels, 1) * inputs.size(0)
            jaccard2 += Jaccard_Coeff(res, labels, 2) * inputs.size(0)
            jaccard3 += Jaccard_Coeff(res, labels, 3) * inputs.size(0)

            sensitive1 += Sensitive_Coeff(res, labels, 1) * inputs.size(0)
            sensitive2 += Sensitive_Coeff(res, labels, 2) * inputs.size(0)
            sensitive3 += Sensitive_Coeff(res, labels, 3) * inputs.size(0)

            accuracy += Accuracy(res, labels) * inputs.size(0)
            
    avg_valid_loss = valid_loss / sets.test_size

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
    train()
    # test()
    
    
