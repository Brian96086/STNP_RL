import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
import datetime
import torch
from engine import sample_z, data_to_z_params

class Game():
    
    def __init__(self, cfg, dcrnn, action_space, scenario_dict, dataset_idx, is_online = False):
        #super(Game, self).__init__()
        self.SAVE_PATH = cfg.DIR.output_dir
        self.device = torch.device(cfg.TRAIN.device)
        self.action_space = action_space
        self.scenario_dict = scenario_dict
        self.reward_penalty = cfg.TRAIN.reward_penalty
        self.MAX_ACTIONS  = cfg.TRAIN.max_actions
        self.cfg = cfg
        self.dcrnn = dcrnn
        
        self.num_agents = 1
        self._agent_ids = list(range(self.num_agents))
        self.actions_made = 0

        self.episode_count = 0
        self.gt_reward_arr = None if "gt_rewards" not in scenario_dict.keys() else scenario_dict["gt_rewards"]
        self.is_online = is_online
        self.dataset_idx = dataset_idx
        self.context_pts = self.scenario_dict["context_pts"][self.dataset_idx:self.dataset_idx+1]
        self.target_pts = self.scenario_dict["target_pts"][self.dataset_idx:self.dataset_idx+1]
        self.latent_variable = self.scenario_dict["latent_variable"][self.dataset_idx:self.dataset_idx+1]
        self.valid_pred = self.scenario_dict["valid_pred"][self.dataset_idx:self.dataset_idx+1]
        self.test_pred = self.scenario_dict["test_pred"][self.dataset_idx:self.dataset_idx+1]
        
    
    def step(self, action_idx):
        idx = self.dataset_idx
        if(self.gt_reward_arr != None): #load from existing data
            self.gt_reward_arr = self.scenario_dict["gt_rewards"][idx:idx+1].reshape(-1,1)
            self.gt_reward_arr = (self.gt_reward_arr - torch.mean(self.gt_reward_arr))/torch.std(self.gt_reward_arr)
            sorted_idx = np.argsort(self.gt_reward_arr.flatten().numpy().copy())
            self.top_twenty_idx = sorted_idx[-20:]
#             self.reward_rank_arr = torch.ones(self.gt_reward_arr.shape)*-1
#             self.reward_rank_arr[self.top_twenty_idx] = 1
            self.reward_dict = {sorted_idx[i]:(270-i) for i in range(len(sorted_idx))} #store the reward rank given an action_idx
            
        else: #online training
            x_c, y_c = torch.split(self.context_pts, [2, 100], dim = 2)
            x_t, y_t, y_t_pred = torch.split(self.target_pts, [2, 100, 100], dim = 2) #n_iter x n_pts x pt_dim
            x_train = torch.cat([x_c, x_t], dim = 1)[0]
            y_train = torch.cat([y_c, y_t], dim = 1)[0]
            self.gt_reward_arr = self.calculate_score(self.cfg, self.dcrnn, x_train, y_train, self.action_space)
        if(self.actions_made == 0):
            print("reward stats: mean = {}, median = {}, std = {}, max = {}, min = {}".format(
                np.round(torch.mean(self.gt_reward_arr),3), 
                np.round(torch.median(self.gt_reward_arr),3), 
                np.round(torch.std(self.gt_reward_arr),3), 
                np.round(torch.max(self.gt_reward_arr),3),
                np.round(torch.min(self.gt_reward_arr),3))
            )
        
        selected_param = self.action_space[action_idx]
        observation = self.dict_to_state(self.context_pts, self.target_pts, self.latent_variable) #convert from dictionary to vector(RL state has to be a flattened vector)
        reward = self.gt_reward_arr[action_idx]*self.reward_penalty**(self.actions_made)
#         reward = torch.tensor([(270-self.reward_dict[action_idx])*self.reward_penalty**(self.actions_made)])
        #reward = self.gt_reward_arr[action_idx]
        self.actions_made +=1
        done = self.actions_made >= self.MAX_ACTIONS
#         if(self.actions_made % 10 ==0):
#             print('data_num = {}, actions_made = {}, reward = {}'.format(self.dataset_idx, self.actions_made, np.round(reward.item(), 3)))
        info = {"selected_param": selected_param, "gt_reward_arr": self.gt_reward_arr.flatten(), "reward_dict":self.reward_dict}
        
        return observation, reward, done, info

    def get_reward(self, action_idx):
        return self.gt_rewards[action_idx]

    
    def calculate_score(self, cfg, dcrnn, x_train, y_train, beta_epsilon_all):
        # query z_mu, z_var of the current training data
        with torch.no_grad():
            z_mu, z_logvar = data_to_z_params(dcrnn, x_train.to(self.device),y_train.to(self.device))
            score_list = []
            for i in range(len(beta_epsilon_all)):
                # generate x_search
                x1 = beta_epsilon_all[i:i+1]
                x_search = np.repeat(x1,cfg.SIMULATOR.num_simulations,axis =0)
                x_search = torch.from_numpy(x_search).float()

                # generate y_search based on z_mu, z_var of current training data
                output_list = []
                for j in range (len(x_search)):
                    zsamples = sample_z(z_mu, z_logvar,cfg.MODEL.z_dim) 
                    output = dcrnn.decoder(x_search[j:j+1].to(self.device), zsamples).cpu()
                    output_list.append(output.detach().numpy())

                y_search = np.concatenate(output_list)
                y_search = torch.from_numpy(y_search).float()

                x_search_all = torch.cat([x_train,x_search],dim=0)
                y_search_all = torch.cat([y_train,y_search],dim=0)

                # generate z_mu_search, z_var_search
                z_mu_search, z_logvar_search = data_to_z_params(dcrnn, x_search_all.to(self.device),y_search_all.to(self.device))
                
                # calculate and save kld
                mu_q, var_q, mu_p, var_p = z_mu_search,  0.1+ 0.9*torch.sigmoid(z_logvar_search), z_mu, 0.1+ 0.9*torch.sigmoid(z_logvar)

                std_q = torch.sqrt(var_q)
                std_p = torch.sqrt(var_p)

                p = torch.distributions.Normal(mu_p, std_p)
                q = torch.distributions.Normal(mu_q, std_q)
                score = torch.distributions.kl_divergence(p, q).sum()

                score_list.append(score.item())
            score_array = np.array(score_list)
        return score_array

    def reset(self):        
        observation = self.dict_to_state(self.context_pts, self.target_pts, self.latent_variable).to(self.device)
        self.actions_made = 0
        return observation
    
    def dict_to_state(self, context_pts, target_pts, latent_variable):
        ct_shape = torch.tensor(context_pts.shape[-2:])
        tgt_shape = torch.tensor(target_pts.shape[-2:])
        context_pts = context_pts.flatten()
        context_pts = (context_pts-torch.mean(context_pts))/(torch.std(context_pts))
        target_pts = target_pts.flatten()
        target_pts = (target_pts-torch.mean(target_pts))/(torch.std(target_pts))
        latent_variable = latent_variable.flatten()
        latent_variable = (latent_variable-torch.mean(latent_variable))/(torch.std(latent_variable))
        
        observation = torch.cat([torch.tensor([self.actions_made]), context_pts,target_pts , latent_variable, ct_shape, tgt_shape], dim = 0).to(self.device)
        return observation
              
    def update_data_idx(self, new_idx):
        self.dataset_idx = new_idx    
        self.context_pts = self.scenario_dict["context_pts"][self.dataset_idx:self.dataset_idx+1]
        self.target_pts = self.scenario_dict["target_pts"][self.dataset_idx:self.dataset_idx+1]
        self.latent_variable = self.scenario_dict["latent_variable"][self.dataset_idx:self.dataset_idx+1]
        self.valid_pred = self.scenario_dict["valid_pred"][self.dataset_idx:self.dataset_idx+1]
        self.test_pred = self.scenario_dict["test_pred"][self.dataset_idx:self.dataset_idx+1]
        self.gt_reward_arr = self.scenario_dict["gt_rewards"][self.dataset_idx:self.dataset_idx+1].reshape(-1,1)
        print('IDX = {}, VAR SHAPE = {}, {}, {}'.format(self.dataset_idx, self.context_pts.shape, self.target_pts.shape, self.latent_variable.shape))
      
        
    
    def render(self, mode):
        pass
    
    def close(self):
        pass
    
    def seed(self):
        pass
