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
    def __init__(self, agent_size, n_params, n_gen_params, args):
        self.agent_data_sizes = agent_size
        self.n_params = n_params
        self.n_gen_params = n_gen_params
        self.args = args
        self.server_lr = 1.0
    
    def aggregate_update(self, global_model, global_generate_model, agent_updates_dict, generated_updates_dict, zda, lam):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(torch.device("cuda:6"))
        lr_gen_vector = torch.Tensor([self.server_lr]*self.n_gen_params).to(torch.device("cuda:6"))

        aggregated_updates = 0
        generate_updates = 0
        klv_loss = 0
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
            generate_updates = self.agg_avg(generated_updates_dict)
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)

        
        temp_model = UNet2D(1, 4, final_sigmoid=False).to(torch.device("cuda:6"))
        temp_generate_model = VAE().to(torch.device("cuda:6"))


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

        
        # loss_function = nn.MSELoss()
        loss_function = nn.KLDivLoss(reduction='batchmean')
        # loss_function2 = nn.MSELoss()
        optimizer1 = torch.optim.Adam(global_model.parameters(), lr=0.0001)
        optimizer2 = torch.optim.Adam(global_generate_model.parameters(), lr=0.0001)

        global_model.train()
        global_generate_model.train()


        # for j in range(10):
        #     get_z = z_data[j]
        #     for k in range(5):
        #         data = get_z[k]
        #         res = torch.zeros(20, 4, 1, 224, 224).to(torch.device("cuda:0"))
        #         for l in range(10):
        #             vector_to_parameters(model_param[l], temp_model.parameters())
        #             vector_to_parameters(generate_param[l], temp_generate_model.parameters())
        #             recons = torch.utils.checkpoint.checkpoint(temp_generate_model.get_recons, data)
        #             recons = recons.view(20, 1, 1, 224, 224)
        #             output = torch.utils.checkpoint.checkpoint(temp_model, recons, data)
        #             res += output
                
        #         res = torch.div(res, 10)
        #         res = res.detach()
        #         recons = global_generate_model.get_recons(data)
        #         recons = recons.view(20, 1, 1, 224, 224)
        #         out_put = global_model(recons, data)

        #         loss = loss_function(out_put, res)
        #         loss = loss * lam
        #         optimizer1.zero_grad()
        #         optimizer2.zero_grad()
        #         loss.backward()
        #         optimizer1.step()
        #         optimizer2.step()

        # num = random.randrange(10)
        # get_z = z_data[num]
        
        klv_loss = 0
        for k in range(2):
            # data = get_z[k]
            data = torch.randn(20, 128).to(torch.device("cuda:6"))
            res = torch.zeros(20, 4, 1, 224, 224).to(torch.device("cuda:6"))
            for l in range(10):
                vector_to_parameters(model_param[l], temp_model.parameters())
                vector_to_parameters(generate_param[l], temp_generate_model.parameters())
                recons = torch.utils.checkpoint.checkpoint(temp_generate_model.get_recons, data, use_reentrant=False)
                recons = recons.view(20, 1, 1, 224, 224)
                output = torch.utils.checkpoint.checkpoint(temp_model, recons, data, use_reentrant=False)
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


        return klv_loss / 40


        # indice = [i for i in range(10)]
        # random.shuffle(indice)
        # indice = indice[:8]
        
        # param = []

        # for i in indice:
        #     temp_global_params = cur_global_params + agent_updates_dict[i]
        #     param.append(temp_global_params.float())
        
        # get_res = []

        # for j in range(len(recons_data)):
        #     data = recons_data[j]
        #     z = z_da[j]
        #     data = torch.div(data, 255)
        #     data = data.view(data.shape[0], 1, 1, data.shape[2], data.shape[3])
        #     label = torch.zeros(12, 4, 1, 224, 224).to(torch.device('cuda:0'))
        #     for k in range(8):
        #         vector_to_parameters(param[k], global_model.parameters())
        #         out_put = torch.utils.checkpoint.checkpoint(global_model, data, z)
        #         label += out_put
        #     label = torch.div(label, 8)
        #     get_res.append(label)
        
        # new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        # vector_to_parameters(new_global_params, global_model.parameters())
        
        # return get_res

            

    def aggregate_generate_update(self, global_model, agent_updates_dict, z_data):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(torch.device("cuda:6"))
        # lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(torch.device("cpu"))
        aggregated_updates = 0
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)

        tmp_model = VAE().to(torch.device('cuda:6'))
        # loss_function = nn.MSELoss()
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01)

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())
        param = []
        for i in range(10):
            temp_global_params = cur_global_params + agent_updates_dict[i]
            param.append(temp_global_params.float())
        
        # num = random.randrange(10)
        # print(num)
        # temp_global_params = cur_global_params + agent_updates_dict[num]
        # vector_to_parameters(temp_global_params.float(), tmp_model.parameters())
        # recons_data = []
        # get_z = z_data[num]
        # for z in get_z:
        #     data = tmp_model.get_recons(z)
        #     recons_data.append(data)

        # get_z_data = []
        # z_da = []
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(global_model.parameters(), lr=0.0001)
        for k in range(5):
            data = torch.randn(20, 128).to(torch.device("cuda:6"))
            res = torch.zeros(20, 1, 224, 224).to(torch.device("cuda:6"))
            for l in range(10):
                vector_to_parameters(param[l], tmp_model.parameters())
                recons_data = tmp_model.get_recons(data)
                res += recons_data
            res = torch.div(res, 10)
            res = res.detach()
            output = global_model.get_recons(get_z[k])

            loss = loss_function(out_put.log(), res)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return 







        # for j in range(5): 
        #     # res = torch.randn(4, 1, 224, 224).to(torch.device('cuda:0'))
        #     data = torch.randn(4, 128).to(torch.device("cuda:0"))
        #     # for k in range(10):
        #     #     vector_to_parameters(param[k], tmp_model.parameters())
        #     #     out_put = tmp_model.get_recons(data)
        #     #     res += out_put
        #     # res = torch.div(res, 10)
        #     # get_z_data.append(res)
        #     get_generate_data = global_model.get_recons(data)
        #     get_z_data.append(get_generate_data)
        #     z_da.append(data)
        #     # res = res.detach()
        #     # output = global_model.get_recons(data)
        #     # loss = loss_function(output, res)

        #     # optimizer.zero_grad()
        #     # loss.backward()
        #     # print(loss.item())
        #     # optimizer.step()

        # new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        # vector_to_parameters(new_global_params, global_model.parameters())

        # return z_data[num], recons_data, re_label[num]     
    
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