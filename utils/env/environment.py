import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
import datetime
import torch
from engine import sample_z, data_to_z_params

class Game():
    
    def __init__(self, cfg, action_space, scenario_dict):
        #super(Game, self).__init__()
        self.SAVE_PATH = cfg.DIR.output_dir
        self.device = torch.device(cfg.TRAIN.device)
        self.action_space = action_space
        self.scenario_dict = scenario_dict
        self.action_made = 0
        self.MAX_ACTIONS  = cfg.TRAIN.max_actions
        
        self.num_agents = 1
        self._agent_ids = list(range(self.num_agents))

        self.A, self.population = self.make_world()
        self.initial_infected = self.make_initial_infected()

        self.ExpPopIn = np.zeros(self.NUM_CITIES)
        self.episode_count = 0
        self.gt_rewards = None
        
    
    def step(self, dcrnn, action_idx):
        if(self.gt_rewards == None):
            context_pts = self.scenario_dict["context_pts"]
            target_pts = self.scenario_dict["latent_variable"]
            valid_pred = self.scenario_dict["valid_pred"]
            test_pred = self.scenario_dict["test_pred"]
            x_c, y_c = self.scenario_dict["context"]
            self.gt_rewards = self.calculate_score(dcrnn)
        selected_param = self.action_space[action_idx]
        observation = self.scenario_dict
        reward = self.gt_rewards[action_idx]
        
        self.action_made +=1

            
        done = {"__all__": self.action_made >= self.max_week,}
        info = {}
        
        return observation, reward, done, info


    def get_reward(self,dcrnn, action_idx):
        return self.compute_acquisition_reward(dcrnn, action_idx)

    def compute_acquistion_reward(self, dcrnn, action_idx):
        return self.step(dcrnn, action_idx)

    
    def calculate_score(self, cfg, dcrnn, x_train, y_train, beta_epsilon_all):
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        # query z_mu, z_var of the current training data
        with torch.no_grad():
            z_mu, z_logvar = data_to_z_params(x_train.to(self.device),y_train.to(self.device))
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
        return {city: self.get_observation(city) for city in range(self.NUM_CITIES)}

    
    def render(self, mode):
        pass
    
    
    def close(self):
        pass

    
    def seed(self):
        pass
