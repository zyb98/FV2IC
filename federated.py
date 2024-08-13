import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import UNet2D
from tqdm import tqdm
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from sklearn import metrics
from setting import parse_opt
import cv2 as cv
import copy
from agent import Agent
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from aggregation import Aggregation
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
import torchsummary
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
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

sets = parse_opt()

def train():
    df = pd.DataFrame(columns=['round', 'loss', 'dice', 'dice1', 'dice2', 'dice3'])
    df.to_csv("ours_valid_results.csv", index=False)
    # df = pd.DataFrame(columns=['round', 'loss', 'dice', 'jaccard', 'sensitive', 'accuracy'])
    # df.to_csv("ours_test_results.csv", index=False)
    sets = parse_opt()
    # train_data = utils.get_data_loader(sets.batch_size, 'train_data')
    valid_data = utils.get_train_data_loader(sets.batch_size, 'final_valid_data')
    # test_data = utils.get_train_data_loader(sets.batch_size, 'exp_test_data3')
    device = torch.device("cuda:6" if torch.cuda.is_available else "cpu")
    # device = torch.device('cpu')
    global_model = UNet2D(sets.in_channels, sets.out_channels, final_sigmoid=False)
    # global_model.load_state_dict(torch.load("ours_100_round_model.pth"))
    global_model = global_model.to(device)
    global_generate_model = VAE()
    # global_generate_model.load_state_dict(torch.load("ours_100_round_generate_model.pth"))
    global_generate_model = global_generate_model.to(device)
    optimizer1 = optim.Adam(global_model.parameters(), sets.learning_rate)
    optimizer2 = optim.Adam(global_model.parameters(), sets.learning_rate)
    agents, agent_data_size, agent_generate_data_size = [], {}, {}
    for agent_id in range(sets.num_agents):
        agent = Agent(agent_id, sets)
        agent_data_size[agent_id] = agent.n_data
        agent_generate_data_size[agent_id] = agent.n_generate_data
        agents.append(agent)
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    n_generate_model_params = len(parameters_to_vector(global_generate_model.parameters()))
    aggregator = Aggregation(agent_data_size, n_model_params, n_generate_model_params, sets)
    # generate_aggregator = Aggregation(agent_generate_data_size, n_generate_model_params, sets)
    # new_generate_aggregator = Aggregation(agent_data_size, n_generate_model_params, sets)
    lr_lambda = lambda epoch:1.0 if epoch<100 else np.math.exp(0.1*(100-epoch))
    # scheduler1 = LambdaLR(optimizer=optimizer1,lr_lambda=lr_lambda)
    # scheduler2 = LambdaLR(optimizer=optimizer2,lr_lambda=lr_lambda)
    best_dice = 0.0

    agent_output_generate_dict = {}
    get_generate_data = []

    z_da = []
    get_label = []
    recons_data = []
    for i in range(5):
        get_z = torch.randn(20, 128).to(device)
        label = torch.randn(20, 4, 1, 224, 224).to(device)
        rec = global_generate_model.get_recons(get_z)
        z_da.append(get_z)
        get_label.append(label)
        recons_data.append(rec)
    

    get_dice_loss = {}
    get_ce_loss = {}
    get_generate_loss = {}
    get_cos_loss = {}
    recons_seg_loss = {}

    klv = []
    # for r in range(10):
    #     get_dice_loss.setdefault(str(r), [])
    #     get_ce_loss.setdefault(str(r), [])
    #     get_generate_loss.setdefault(str(r), [])
    #     get_cos_loss.setdefault(str(r), [])



    for rnd in range(1, sets.rounds + 1):
        # lam = 0.005 * rnd
        lam = 1 / (np.exp(40 - rnd) + 1)
        print(rnd, lam)
        epoch_start = time.time()
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        rnd_generate_global_params = parameters_to_vector(global_generate_model.parameters()).detach()
        agent_updates_dict = {}
        agent_generate_updates_dict = {}
        # for agent_id in range(sets.num_agents):
        #     # print(agent_id)
        #     generate_update = agents[agent_id].train_generate_model(global_generate_model)
        #     agent_generate_updates_dict[agent_id] = generate_update
        #     # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
        #     vector_to_parameters(copy.deepcopy(rnd_generate_global_params), global_generate_model.parameters())
        # # aggregate params obtained by agents and update the global params
        # generate_aggregator.aggregate_generate_update(global_generate_model, agent_generate_updates_dict)


        # res = []
        # count = 0
        # for i in range(10):
        #     pth = os.path.join("generate_data_small", "generate_data_" + str(i))
        #     img = os.listdir(pth)
        #     for j in img:
        #         get_img = Image.open(os.path.join("generate_data_small", "generate_data_" + str(i), j))
        #         get_img = np.array(get_img)
        #         get_img = torch.tensor(get_img, dtype=torch.float32).to(device)
        #         get_img = get_img.view(1, 1, get_img.shape[0], get_img.shape[1])
        #         _, _, _, z = global_generate_model(get_img)
        #         z = list(np.array(z.detach().cpu()))
        #         res.append(z)
        # res = np.array(res)
        # res = np.reshape(res, (899, 128))
        # mean_value = np.mean(res, axis=0)
        # std_value = np.std(res, axis=0)

        get_generate_data  = []


        # for generate_id in range(4):
        #     output = global_generate_model.sample(device)
        #     output = output[0].detach().cpu()
        #     # img = Image.open('VAEResult/' + str(generate_id + 1) + '.png')
        #     # new_output = np.array(img)
        #     # new_output = new_output.view(224, 224)
        #     output = output.view(224, 224)
        #     new_output = np.array(output)
        #     cv.imwrite('VAEResult/generate_' + str(generate_id) + '.png', new_output)

            # img = ToPILImage()(output)
            # img.save('VAEResult/generate_' + str(generate_id) + '.png')

            # img = cv.imread('VAEResult/generate_' + str(generate_id) + '.jpg', cv.IMREAD_COLOR)
            # img = cv.resize(img, (224, 224))
            # img = img.transpose(2, 0, 1)
            # img = np.array(img)

        #     new_img = list(new_output)
        #     get_generate_data.append(new_img)
        # get_generate_data = np.array(get_generate_data)
        # get_generate_data = torch.tensor(get_generate_data, dtype=torch.float32)
        # get_generate_data = get_generate_data.view(get_generate_data.shape[0], 1, 1, get_generate_data.shape[1],  get_generate_data.shape[2])
        # get_generate_data = torch.div(get_generate_data, 255)
        # get_generate_data = get_generate_data.to(device)


        z_data = {}
        re_label = {}


        for agent_id in tqdm(range(sets.num_agents)):
            # print(agent_id)
            update1, update2, diceloss, celoss, generateloss, cosloss = agents[agent_id].train_local(global_model, global_generate_model, lam)
            # update1 = agents[agent_id].train_local(global_model, global_generate_model)
            # update = agents[agent_id].train_local(global_model, get_generate_data, global_generate_model)

            # make sure every agent gets same copy of the global model in a round (i.e., they don't affect each other's training)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
            vector_to_parameters(copy.deepcopy(rnd_generate_global_params), global_generate_model.parameters())

            agent_updates_dict[agent_id] = update1
            agent_generate_updates_dict[agent_id] = update2
            z_data[agent_id] = z_da
            get_dice_loss.setdefault(agent_id, []).append(diceloss.detach().cpu())
            get_ce_loss.setdefault(agent_id, []).append(celoss.detach().cpu())
            get_generate_loss.setdefault(agent_id, []).append(generateloss.detach().cpu())
            get_cos_loss.setdefault(agent_id, []).append(cosloss.detach().cpu())
            # recons_seg_loss.setdefault(agent_id, []).append(reconsegloss.detach().cpu())

        
        plt_dice = np.array(get_dice_loss[5])
        plt_ce = np.array(get_ce_loss[5])
        plt_generate = np.array(get_generate_loss[5])
        plt_cos = np.array(get_cos_loss[5])
        # plt_recons = np.array(recons_seg_loss[0])
        np.save("plt_dice5.npy", plt_dice)
        np.save("plt_ce5.npy", plt_ce)
        np.save("plt_generate5.npy", plt_generate)
        np.save("plt_cos5.npy", plt_cos)
        # np.save("recons_seg_loss.npy", plt_recons)
        
        # for agent_id in tqdm(range(sets.num_agents)):
        #     # print(agent_id)
        #     # generate_data = agents[agent_id].train_local(global_model, get_generate_data)
        #     vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
        #     update = agents[agent_id].train_generate_data(agent_output_generate_dict, global_model, get_generate_data, lam)
        #     # update = agents[agent_id].intra_client(agent_updates_dict, global_model)
        #     agent_updates_dict[agent_id] = update

        # aggregate params obtained by agents and update the global params
        # new_generate_aggregator.aggregate_generate_update(global_generate_model, agent_generate_updates_dict)
        # z_da, recons_data, new_ret_label = generate_aggregator.aggregate_generate_update(global_generate_model, agent_generate_updates_dict, z_data, re_label)
        # img_gen = recons_data[0][0].detach().cpu()
        # img_gen = img_gen.view(224, 224)
        # img_gen = np.array(img_gen)
        # cv.imwrite('gen_img.png', img_gen)
        # recons_data = []
        # for n in range(len(z_da)):
        #     data = global_generate_model.get_recons(z_da[n])
        #     recons_data.append(data)
        # generate_aggregator.aggregate_update(global_generate_model, agent_generate_updates_dict)
        # n = random.randrange(10)
        # zda = z_data[n]
        klv_loss = aggregator.aggregate_update(global_model, global_generate_model, agent_updates_dict, agent_generate_updates_dict, z_data, lam)
        # klv.append(klv_loss.detach().cpu())
        # plt_klv = np.array(klv)
        # np.save("plt_klv.npy", plt_klv)
        
        # get_label = new_ret_label
        # new_get_label = get_label[0][0].detach().cpu()
        # _, get_pre = torch.max(new_get_label, 0)
        # get_pre = get_pre.view(224, 224)
        # get_pre = get_pre * 85
        # get_pre = np.array(get_pre)
        # cv.imwrite("gen_label.png", get_pre)



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
                # outputs = global_model(inputs)
                loss = total_dice_loss(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, pre = torch.max(outputs, 1)
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

                # jaccard1 += Jaccard_Coeff1(res, labels) * inputs.size(0)
                # jaccard2 += Jaccard_Coeff2(res, labels) * inputs.size(0)
                # jaccard3 += Jaccard_Coeff3(res, labels) * inputs.size(0)

                # sensitive1 += Sensitive_Coeff1(res, labels) * inputs.size(0)
                # sensitive2 += Sensitive_Coeff2(res, labels) * inputs.size(0)
                # sensitive3 += Sensitive_Coeff3(res, labels) * inputs.size(0)

                # accuracy += Accuracy(res, labels) * inputs.size(0)
        
        avg_valid_loss = valid_loss / sets.valid_size
        # avg_valid_dice = valid_dice / sets.test_size
        avg_valid_dice1 = dice1 / sets.valid_size
        avg_valid_dice2 = dice2 / sets.valid_size
        avg_valid_dice3 = dice3 / sets.valid_size
        avg_valid_dice = (avg_valid_dice1 + avg_valid_dice2 + avg_valid_dice3) / 3

        # avg_valid_jaccard1 = jaccard1 / sets.valid_size
        # avg_valid_jaccard2 = jaccard2 / sets.valid_size
        # avg_valid_jaccard3 = jaccard3 / sets.valid_size
        # avg_valid_jaccard = (avg_valid_jaccard1 + avg_valid_jaccard2 + avg_valid_jaccard3) / 3

        # avg_valid_sensitive1 = sensitive1 / sets.valid_size
        # avg_valid_sensitive2 = sensitive2 / sets.valid_size
        # avg_valid_sensitive3 = sensitive3 / sets.valid_size
        # avg_valid_sensitive = (avg_valid_sensitive1 + avg_valid_sensitive2 + avg_valid_sensitive3) / 3

        # avg_valid_accuracy = accuracy / sets.valid_size





        # result = global_model(test_img)
        # res1 = torch.where(result[:, 1, :, :, :]>0.5, torch.tensor(85).to(device), torch.tensor(0).to(device))
        # res2 = torch.where(result[:, 2, :, :, :]>0.5, torch.tensor(170).to(device), torch.tensor(0).to(device))
        # res3 = torch.where(result[:, 3, :, :, :]>0.5, torch.tensor(255).to(device), torch.tensor(0).to(device))
        # res1 = res1.view(224, 224)
        # res2 = res2.view(224, 224)
        # res3 = res3.view(224, 224)
        # res = res1 + res2 + res3

        # print(res.shape)
        # _, get_pre = torch.max(result, 1)
        # get_pre = get_pre.view(224, 224)
        # get_pre = torch.mul(get_pre, 85)

        if best_dice < avg_valid_dice:
            best_dice = avg_valid_dice
            torch.save(global_model.state_dict(), "ours_global_model_5%.pth")
            torch.save(global_generate_model.state_dict(), "ours_global_generate_model_5%.pth")
            # get_pre = np.array(res.cpu())
            # cv.imwrite("new_get_label.png", get_pre)
        


        # scheduler1.step()
        # scheduler2.step()
        epoch_end = time.time()
        print("Round: {:03d}, Validation: Loss: {:.4f} Dice: {:4f}, Dice1: {:.4f}, Dice2: {:.4f}, Dice3: {:.4f} Time: {:.4f}s".format(
            rnd, avg_valid_loss, avg_valid_dice, avg_valid_dice1, avg_valid_dice2, avg_valid_dice3, epoch_end-epoch_start
        ))
        print("Best Dice:", best_dice.item())
        ndf = pd.DataFrame(columns=["round", "loss", "dice", "dice1", "dice2", "dice3"])
        ndf.loc[0] = [rnd] + [np.array(avg_valid_loss)] + [np.array(avg_valid_dice.cpu())] + [np.array(avg_valid_dice1.cpu())] + [np.array(avg_valid_dice2.cpu())] + [np.array(avg_valid_dice3.cpu())]
        ndf.to_csv("ours_valid_results.csv", mode="a", index=False, header=False)

        # with open("get_dice.json", "w") as fp:
        #     json.dump(get_dice_loss.cpu(), fp)
        
        # with open("get_ce.json", "w") as fp:
        #     json.dump(get_ce_loss.cpu(), fp)

        # with open("get_cos", "w") as fp:
        #     json.dump(get_cos_loss.cpu(), fp)



    # torch.save(global_generate_model.state_dict(), "global_generate_model.pth")

        
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


# def test():
#     device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#     global_model = UNet2D(1, 4, final_sigmoid=False)
#     global_generate_model = VAE()
#     global_model.load_state_dict(torch.load("ours_global_model_5%.pth"))
#     global_generate_model.load_state_dict(torch.load("ours_global_generate_model_5%.pth"))
#     global_model = global_model.to(device)
#     global_generate_model = global_generate_model.to(device)
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
#         global_model.eval()
#         global_generate_model.eval()

#         for j, (inputs, labels) in enumerate(test_data):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             new_inputs = inputs.view(inputs.shape[0], 1, inputs.shape[3], inputs.shape[4])
#             recons_data, _, _, z = global_generate_model(new_inputs)
#             outputs = global_model(inputs, z)
#             # outputs = global_model(inputs)
                
#             loss = total_dice_loss(outputs, labels)
#             valid_loss += loss.item() * inputs.size(0)
#             _, pre = torch.max(outputs, 1)
#             res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
#             res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
#             res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
#             res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
#             res = torch.cat((res1, res2, res3, res4), 1)
#             res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])
#             # res = torch.where(outputs>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
#             # valid_dice += Dice_Coeff(res, labels) * inputs.size(0)
#             dice1 += Dice_Coeff1(res, labels) * inputs.size(0)
#             dice2 += Dice_Coeff2(res, labels) * inputs.size(0)
#             dice3 += Dice_Coeff3(res, labels) * inputs.size(0)

#             jaccard1 += Jaccard_Coeff1(res, labels) * inputs.size(0)
#             jaccard2 += Jaccard_Coeff2(res, labels) * inputs.size(0)
#             jaccard3 += Jaccard_Coeff3(res, labels) * inputs.size(0)

#             sensitive1 += Sensitive_Coeff1(res, labels) * inputs.size(0)
#             sensitive2 += Sensitive_Coeff2(res, labels) * inputs.size(0)
#             sensitive3 += Sensitive_Coeff3(res, labels) * inputs.size(0)

#             accuracy += Accuracy(res, labels) * inputs.size(0)



            
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
#         avg_valid_loss, avg_valid_dice, avg_valid_jaccard, avg_valid_sensitive, avg_valid_accuracy))


def add_color():
    img = cv.imread("initial_image_next.png", cv.IMREAD_COLOR)
    label_img = cv.imread("PLD_test_next.png", cv.IMREAD_GRAYSCALE)
    for i in range(224):
        for j in range(224):
            if label_img[i][j] == 85:
                img[i][j][0] = 205
                img[i][j][1] = 224
                img[i][j][2] = 64
            if label_img[i][j] == 170:
                img[i][j][0] = 0
                img[i][j][1] = 128
                img[i][j][2] = 255
            if label_img[i][j] == 255:
                img[i][j][0] = 255
                img[i][j][1] = 0
                img[i][j][2] = 0
    cv.imwrite("add_color_PLD_next.png", img)



def test():
    device = torch.device("cuda:7" if torch.cuda.is_available else "cpu")
    global_model = UNet2D(1, 4, final_sigmoid=False)
    global_generate_model = VAE()
    global_model.load_state_dict(torch.load("ours_global_model_5%.pth"))
    global_generate_model.load_state_dict(torch.load("ours_global_generate_model_5%.pth"))
    global_model = global_model.to(device)
    global_generate_model = global_generate_model.to(device)
    test_data = utils.get_train_data_loader(sets.batch_size, 'final_valid_data')

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
    # test()
    # train_1 = np.load("new_upper1/train_data_0/1.npy")
    # print(train_1)

    # for i in range(10):
    #     for j in range(61, 101):
    #         img_pth = "new_upper60%/train_data_" + str(i) + "/" + str(j) + ".npy"
    #         label_pth = "new_upper60%/train_label_" + str(i) +"/" + str(j) + ".npy"
    #         os.remove(img_pth)
    #         os.remove(label_pth)


    # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    # model = UNet2D(1, 4, final_sigmoid=False)
    # generate_model = VAE()
    # model.load_state_dict(torch.load("ours_new_global_model.pth"))
    # generate_model.load_state_dict(torch.load("ours_new_global_generate_model.pth"))
    # model = model.to(device)
    # generate_model = generate_model.to(device)
    # img = np.load("final_test_data/12.npy")
    # label_img = np.load("final_test_label/12.npy")
    # img = torch.tensor(img, dtype=torch.float32)
    # img = img.to(device)
    # img = img.view(1, 1, 224, 224)
    # _, _, _, z = generate_model(img)
    # img = img.view(1, 1, 1, 224, 224)
    # label = torch.tensor(label_img, dtype=torch.float32)
    # label = label.to(device)
    # label = label.view(label.shape[0] * label.shape[1], 1)
    # label = np.array(label.cpu())
    # y, fea = model(img, z)
    # # print(outputs[0].shape)
    # fea = fea.view(fea.shape[1], fea.shape[3] * fea.shape[4])
    # fea = np.array(fea.detach().cpu())
    # fea = np.transpose(fea, (1, 0))
    # pca = PCA(n_components=2)
    # new_fea = pca.fit_transform(fea)
    # x1 = []
    # y1 = []
    # x2 = []
    # y2 = []
    # x3 = []
    # y3 = []
    # for i in label.shape[0]:
    #     if label[i][0] == 85:
    #         x1.append(new_fea[i][0])
    #         y1.append(new_fea[i][1])
    #     if label[i][0] == 170:
    #         x2.append(new_fea[i][0])
    #         y2.append(new_fea[i][1])
    #     if label[i][0] == 255:
    #         x3.append(new_fea[i][0])
    #         y3.append(new_fea[i][1])
    
    # plt.figure()
    # plt.scatter(x1, y1, c='red')
    # plt.scatter(x2, y2, c='black')
    # plt.scatter(x3, y3, c='orange')

    # plt.show()
    

        


    





    # add_color()
    # img = cv.imread("initial_image_next.png", cv.IMREAD_COLOR)
    # cv.imwrite("new_initial_image_next.png", img)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = UNet2D(1, 4, final_sigmoid=False)
    # model.load_state_dict(torch.load("ours_new_global_model.pth"))
    model.load_state_dict(torch.load("PLD_model.pth"))
    model = model.to(torch.device("cuda:0"))
    generate_model = VAE()
    generate_model.load_state_dict(torch.load("ours_new_global_generate_model.pth"))
    generate_model = generate_model.to(torch.device("cuda:0"))
    model.eval()
    generate_model.eval()
    # all_dice = 0.0
    # df = pd.DataFrame(columns=['num', 'dice'])
    # df.to_csv("FedIRM_statistic.csv", index=False)
    test_img = np.load("final_test_data/145.npy")
    label_img = np.load("final_test_label/145.npy")
    cv.imwrite("new_image.png", np.round(test_img * 255))
    test_img = torch.tensor(test_img, dtype=torch.float32)
    test_img = test_img.to(torch.device("cuda:0"))
    # test_img = test_img.view(1, 1, 224, 224)
    # _, _, _, z = generate_model(test_img)
    test_img = test_img.view(1, 1, 1, 224, 224)
    # output = model(test_img, z)
    output = model(test_img)
    _, pre = torch.max(output, 1)
    res1 = torch.where(pre==0, torch.tensor(0).to(device), torch.tensor(0).to(device))
    res2 = torch.where(pre==1, torch.tensor(85).to(device), torch.tensor(0).to(device))
    res3 = torch.where(pre==2, torch.tensor(170).to(device), torch.tensor(0).to(device))
    res4 = torch.where(pre==3, torch.tensor(255).to(device), torch.tensor(0).to(device))
    res1 = res1[0][0].detach().cpu()
    res2 = res2[0][0].detach().cpu()
    res3 = res3[0][0].detach().cpu()
    res4 = res4[0][0].detach().cpu()
    res = res1 + res2 + res3 + res4
    res = np.array(res)
    cv.imwrite("PLD_test_next.png", res)  
    cv.imwrite("new_test_label.png", label_img) 



    # for i in range (281):
    #     test_img = np.load("final_test_data/" + str(i + 1) + ".npy")
    #     label_img = np.load("final_test_label/" + str(i + 1) + ".npy")

    #     test_img = torch.tensor(test_img, dtype=torch.float32)
    #     test_img = test_img.to(torch.device("cuda:0"))
    #     test_img = test_img.view(1, 1, 224, 224)
    #     # _, _, _, z = generate_model(test_img)
    #     test_img = test_img.view(1, 1, 1, 224, 224)
    #     # output = model(test_img, z)
    #     output = model(test_img)
    #     _, pre = torch.max(output, 1)
    #     res1 = torch.where(pre==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     res2 = torch.where(pre==1, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     res3 = torch.where(pre==2, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     res4 = torch.where(pre==3, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     res = torch.cat((res1, res2, res3, res4), 1)
    #     res = res.view(res.shape[0], res.shape[1], 1, res.shape[2], res.shape[3])
    #     label = torch.tensor(label_img, dtype=torch.float32)
    #     label = label.to(device)
    #     label = label.view(1, 1, label.shape[0], label.shape[1])
    #     label1 = torch.where(label==0, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     label2 = torch.where(label==85, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     label3 = torch.where(label==170, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     label4 = torch.where(label==255, torch.tensor(1).to(device), torch.tensor(0).to(device))
    #     new_label = torch.cat((label1, label2, label3, label4))
    #     new_label = new_label.view(1, new_label.shape[0], new_label.shape[1], new_label.shape[2], new_label.shape[3])
    #     # print(res.shape)
    #     # print(new_label.shape)
    #     dice1 = Dice_Coeff1(res, new_label)
    #     dice2 = Dice_Coeff2(res, new_label)
    #     dice3 = Dice_Coeff3(res, new_label)
    #     avg_dice = (dice1 + dice2 + dice3) / 3
    #     print(avg_dice.item())
    #     ndf = pd.DataFrame(columns=["num", "dice"])
    #     ndf.loc[0] = [i+1] + [avg_dice.item()]
    #     ndf.to_csv("FedIRM_statistic.csv", mode='a', index=False, header=False)
        # print(res1.shape)
        # res1 = res1[0][0].detach().cpu()
        # res2 = res2[0][0].detach().cpu()
        # res3 = res3[0][0].detach().cpu()
        # res4 = res4[0][0].detach().cpu()
        # res = res1 + res2 + res3 + res4
        # res = np.array(res)
        # cv.imwrite("test.png", res)  
        # cv.imwrite("test_label.png", label_img) 
    




    # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    # generate_model = VAE().to(torch.device('cuda:0'))
    # print(torchsummary.summary(generate_model.cuda(), (1, 224, 224)))

    # model = UNet2D(1, 4, final_sigmoid=False).to(torch.device('cuda:0'))
    # print(torchsummary.summary(model.cuda(), (1, 1, 224, 224)))

    # generate_model.load_state_dict(torch.load("global_generate_model.pth"))
    # output = generate_model.sample(device)
    # output = output[0].detach().cpu()
    # output = output.view(224, 224)
    # new_output = np.array(output)
    # cv.imwrite('test.png', new_output)

    # img = Image.open("exp_test_data2/78.png")
    # img = np.load("exp_test_data3/149.npy")
    # ori_img = img * 255
    # cv.imwrite("original_img.png", ori_img)
    # # img = np.array(img)
    # img = torch.tensor(img, dtype=torch.float32).to(torch.device("cuda:0"))
    # # img = torch.div(img, 255)
    # img = img.view(1, 1, img.shape[0], img.shape[1])
    # generate_model = VAE().to(torch.device("cuda:0"))
    # model = UNet2D(1, 4, final_sigmoid=False).to(torch.device("cuda:0"))
    # generate_model.load_state_dict(torch.load("global_generate_model.pth"))
    # model.load_state_dict(torch.load("global_model.pth"))
    # model.eval()
    # generate_model.eval()
    # y, _, _, z = generate_model(img)
    # y = y[0].detach().cpu().view(224, 224)
    # y = y * 255
    # y = np.array(y)
    # cv.imwrite("recons.png", y)
    # img = img.view(img.shape[0], 1, img.shape[1], img.shape[2], img.shape[3])
    # # y = y.view(y.shape[0], 1, y.shape[1], y.shape[2], y.shape[3])
    # output = model(img, z)
    # # res = torch.where(output>0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
    # _, pre = torch.max(output, 1)
    # res1 = torch.where(pre==0, torch.tensor(0).to(device), torch.tensor(0).to(device))
    # res2 = torch.where(pre==1, torch.tensor(85).to(device), torch.tensor(0).to(device))
    # res3 = torch.where(pre==2, torch.tensor(170).to(device), torch.tensor(0).to(device))
    # res4 = torch.where(pre==3, torch.tensor(255).to(device), torch.tensor(0).to(device))
    # res1 = res1.detach().cpu()
    # res2 = res2.detach().cpu()
    # res3 = res3.detach().cpu()
    # res4 = res4.detach().cpu()
    # # res1 = res[0][0].detach().cpu()
    # # res2 = res[0][1].detach().cpu()
    # # res3 = res[0][2].detach().cpu()
    # # res4 = res[0][3].detach().cpu()
    # res1 = res1.view(224, 224)
    # res2 = res2.view(224, 224)
    # res3 = res3.view(224, 224)
    # res4 = res4.view(224, 224)
    # # res = res1 * 0 + res2 * 85 + res3 * 170 + res4 * 255
    # res = res1 + res2 + res3 + res4
    # res = np.array(res)
    # cv.imwrite("test.png", res)   

    # label_img = np.load("exp_test_label3/149.npy")
    # cv.imwrite("test_label.png", label_img)

    # img = Image.open("test.png")
    # img = np.array(img)
    # res = []
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j] not in res:
    #             res.append(img[i][j])
    # print(res)




    # res = []
    # count = 0
    # for i in range(10):
    #     pth = os.path.join("generate_data_small", "generate_data_" + str(i))
    #     img = os.listdir(pth)
    #     for j in img:
    #         get_img = Image.open(os.path.join("generate_data_small", "generate_data_" + str(i), j))
    #         get_img = np.array(get_img)
    #         get_img = torch.tensor(get_img, dtype=torch.float32)
    #         get_img = get_img.view(1, 1, get_img.shape[0], get_img.shape[1])
    #         _, mean_vec, logvar = generate_model(get_img)
    #         eps = torch.randn_like(logvar)
    #         std = torch.exp(logvar / 2)
    #         z = std * eps + mean_vec
    #         z = list(np.array(z.detach().cpu()))
    #         res.append(z)
    # res = np.array(res)
    # res = np.reshape(res, (899, 128))
    # mean_value = np.mean(res, axis=0)
    # std_value = np.std(res, axis=0)
    # # print(std_value.shape)
    # # print(mean_value.shape)
    # np.save("mean_value.npy", mean_value)
    # np.save("std_value.npy", std_value)

    # test()
    # generate_model = VAE()
    # generate_model.load_state_dict(torch.load("global_generate_model.pth"))
    # pic1 = Image.open("generate_data_small/generate_data_0/22.png")
    # pic1 = np.array(pic1)
    # pic1 = torch.tensor(pic1, dtype=torch.float32)
    # pic1 = pic1.view(1, 1, pic1.shape[0], pic1.shape[1])
    # output = generate_model(pic1)
    # output = output[0].detach().cpu()
    # output = output.view(224, 224)
    # new_output = np.array(output)
    # cv.imwrite('obtain.png', new_output)




    # count = 1

    # for i in range(143, 286):
    #     img = cv.imread("test_data_100/" + str(i) + ".png")
    #     label = cv.imread("test_label_100/" + str(i) + ".png")
    #     cv.imwrite("valid_data_100/" + str(count) + ".png", img)
    #     cv.imwrite("valid_label_100/" + str(count) + ".png", label)
    #     count += 1

    # count = 0
    # num = []
    # img = cv.imread("semi_federated_data/train_label_8/6.png", cv.IMREAD_GRAYSCALE)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j] not in num:
    #             num.append(img[i][j])
    
    # print(num)
    
    # print(count)

    # img = cv.imread("new_semi_federated_data/train_data_0/15.png", cv.IMREAD_COLOR)
    # label = cv.imread("new_semi_federated_data/train_label_0/15.png", cv.IMREAD_GRAYSCALE)
    # print(img.shape)
    # print(label.shape)
    # label = cv.resize(label, (224, 224))
    # num = []
    # for i in range(label.shape[0]):
    #     for j in range(label.shape[1]):
    #         if label[i][j] not in num:
    #             num.append(label[i][j])
    # print(num)

    # print(img.shape)
    # print(label.shape)


    # img = cv.imread("new_generate_data/generate_data_0/19.png", cv.IMREAD_COLOR)
    # img = cv.resize(img, (224, 224))
    # img = img.transpose(2, 0, 1)
    # img = torch.tensor(img, dtype=torch.float32)
    # img = torch.div(img, 255)
    # img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
    # generate_model = VAE()
    # generate_model.load_state_dict(torch.load("global_generate_model.pth"))
    # output = generate_model(img)
    # output = output[0][0].detach().cpu()
    # print(output.shape)
    # img = ToPILImage()(output)
    # img.save("new_img.jpg")

    # print(torch.cuda.is_available())
    # label= cv.imread("train_label/1.png", cv.IMREAD_GRAYSCALE)
    # label = torch.tensor(label)
    # label = torch.div(label, 255, rounding_mode='trunc')
    # for i in range(label.shape[0]):
    #     for j in range(label.shape[1]):
    #         if label[i][j].item() != 0:
    #             print(label[i][j].item())

    # num = []
    # img = cv.imread("label/16.png", cv.IMREAD_GRAYSCALE)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i][j] not in num:
    #             num.append(img[i][j])
    # print(num)
            
    
    
