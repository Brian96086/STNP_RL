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
        self.action_made = 0
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
        
    
    def step(self, action_idx):
        idx = self.dataset_idx
        self.context_pts = self.scenario_dict["context_pts"][idx:idx+1]
        self.target_pts = self.scenario_dict["target_pts"][idx:idx+1]
        self.latent_variable = self.scenario_dict["latent_variable"][idx:idx+1]
        self.valid_pred = self.scenario_dict["valid_pred"][idx:idx+1]
        self.test_pred = self.scenario_dict["test_pred"][idx:idx+1]
        if(self.gt_reward_arr != None): #load from existing data
            self.gt_reward_arr = self.scenario_dict["gt_rewards"][idx:idx+1].reshape(-1,1)
        else: #online training
            x_c, y_c = torch.split(self.context_pts, [2, 100], dim = 2)
            x_t, y_t, y_t_pred = torch.split(self.target_pts, [2, 100, 100], dim = 2) #n_iter x n_pts x pt_dim
            x_train = torch.cat([x_c, x_t], dim = 1)[0]
            y_train = torch.cat([y_c, y_t], dim = 1)[0]
            self.gt_reward_arr = self.calculate_score(self.cfg, self.dcrnn, x_train, y_train, self.action_space)
        if(self.action_made == 1):
            print('max action = {}'.format(self.MAX_ACTIONS))
            torch.sort(self.gt_reward_arr)
            print("reward stats: mean = {}, median = {}, max = {}, top 5 values = {}".format(
                torch.mean(self.gt_reward_arr), torch.median(self.gt_reward_arr), torch.max(self.gt_reward_arr),  self.gt_reward_arr[-5:]
            ))
        
        selected_param = self.action_space[action_idx]
        #observation = self.scenario_dict
        ct_shape = torch.tensor(self.context_pts.shape[-2:])
        tgt_shape = torch.tensor(self.target_pts.shape[-2:])
        observation = torch.cat([self.context_pts.flatten(), self.target_pts.flatten(), self.latent_variable.flatten(),ct_shape , tgt_shape], dim = 0)
        reward = self.gt_reward_arr[action_idx]
        self.action_made +=1
        #done = {"__all__": self.action_made >= self.MAX_ACTIONS}
        done = self.action_made >= self.MAX_ACTIONS
        print('data_num = {}, actions_made = {}, action = {}, reward = {}'.format(self.dataset_idx, self.actions_made, action_idx, reward))
        print('done = ', done)
        info = {"selected_param": selected_param, "gt_reward_arr": self.gt_reward_arr}
        
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
        idx = self.dataset_idx
        self.context_pts = self.scenario_dict["context_pts"][idx:idx+1]
        self.target_pts = self.scenario_dict["target_pts"][idx:idx+1]
        self.latent_variable = self.scenario_dict["latent_variable"][idx:idx+1]
        self.valid_pred = self.scenario_dict["valid_pred"][idx:idx+1]
        self.test_pred = self.scenario_dict["test_pred"][idx:idx+1]
        
        ct_shape = torch.tensor(self.context_pts.shape[-2:])
        tgt_shape = torch.tensor(self.target_pts.shape[-2:])
        observation = torch.cat([self.context_pts.flatten(), self.target_pts.flatten(), self.latent_variable.flatten(),ct_shape, tgt_shape], dim = 0)
        print('reset game - observation shape = {}'.format(observation.shape))
        
        
        self.action_made = 0
        
        
        return observation
    
    def render(self, mode):
        pass
    
    def close(self):
        pass
    
    def seed(self):
        pass
