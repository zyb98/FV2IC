import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from vae import VAE
import torch.nn as nn
import random
from model import UNet2D

class Aggregation():
    def __init__(self, agent_size, n_params, n_gen_params):
        self.agent_data_sizes = agent_size
        self.n_params = n_params
        self.n_gen_params = n_gen_params
        self.server_lr = 1.0
    
    def aggregate_update(self, global_model, global_generate_model, agent_updates_dict, generated_updates_dict, lam):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(torch.device("cuda:0"))
        lr_gen_vector = torch.Tensor([self.server_lr]*self.n_gen_params).to(torch.device("cuda:0"))

        aggregated_updates = 0
        generate_updates = 0
        klv_loss = 0 

        aggregated_updates = self.agg_avg(agent_updates_dict)
        generate_updates = self.agg_avg(generated_updates_dict)

        
        temp_model = UNet2D(1, 4, final_sigmoid=False).to(torch.device("cuda:0"))
        temp_generate_model = VAE().to(torch.device("cuda:0"))


        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())

        cur_generate_params = parameters_to_vector(global_generate_model.parameters())
        new_generate_params =  (cur_generate_params + lr_gen_vector*generate_updates).float() 
        vector_to_parameters(new_generate_params, global_generate_model.parameters())

        model_param = []
        for i in range(10):
            temp_global_params = cur_global_params + agent_updates_dict[i]
            model_param.append(temp_global_params.float())
        
        generate_param = []
        for i in range(10):
            temp_generate_params = cur_generate_params + generated_updates_dict[i]
            generate_param.append(temp_generate_params.float()) 
       
        loss_function = nn.KLDivLoss(reduction='batchmean')

        optimizer1 = torch.optim.Adam(global_model.parameters(), lr=0.0001)
        optimizer2 = torch.optim.Adam(global_generate_model.parameters(), lr=0.0001)

        global_model.train()
        global_generate_model.train()

        for k in range(5):
            data = torch.randn(20, 128).to(torch.device("cuda:0"))
            res = torch.zeros(20, 4, 1, 224, 224).to(torch.device("cuda:0"))
            for l in range(10):
                vector_to_parameters(model_param[l], temp_model.parameters())
                vector_to_parameters(generate_param[l], temp_generate_model.parameters())
                recons = torch.utils.checkpoint.checkpoint(temp_generate_model.get_recons, data)
                recons = recons.view(20, 1, 1, 224, 224)
                output = torch.utils.checkpoint.checkpoint(temp_model, recons, data)
                res += output
            
            res = torch.div(res, 10)
            res = res.detach()
            recons = global_generate_model.get_recons(data)
            recons = recons.view(20, 1, 1, 224, 224)
            out_put = global_model(recons, data)

            loss = loss_function(out_put.log(), res)
            klv_loss += loss * 20
            loss = loss * lam
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

        return
    

    def agg_avg(self, agent_updates_dict):
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data

            

   
