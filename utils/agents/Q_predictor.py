from torch import nn
import numpy as np
import torch
from einops import rearrange

class Q_predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encode_p = nn.Sequential(
            nn.Linear(110, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 384),
            nn.Sigmoid(),
            nn.Linear(384, 270)
        )
        
        #old q
        self.encode_q = nn.Sequential(
            nn.Linear(111, 192),
            nn.LeakyReLU(),
            nn.Linear(192, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 270)
        )
    
    def forward(self, x):
        x = x.reshape(-1,1)
        
        action_token, state_arr, ct_shape, tgt_shape  = torch.split(x, [1, x.shape[0]-5, 2, 2], dim = 0)
        ct_n, ct_dim = int(ct_shape[0]), int(ct_shape[1]) #context
        tgt_n, tgt_dim = int(tgt_shape[0]), int(tgt_shape[1]) #target
        z_length  = state_arr.shape[0]-ct_n*ct_dim-tgt_n*tgt_dim
        ct_arr, tgt_arr, z_arr = torch.split(state_arr.unsqueeze(0), [ct_n*ct_dim, tgt_n*tgt_dim, z_length], dim = 1)
        c_pts = ct_arr.reshape(ct_n, 1, ct_dim)
        t_pts = tgt_arr.reshape(tgt_n, 1, tgt_dim)
        z = z_arr.reshape(1, z_length)
        
        x_c, y_c = torch.split(c_pts, [2,100], dim =2)
        x_t, y_t, y_t_pred = torch.split(t_pts, [2, 100,100], dim = 2) 
        t_pts = torch.cat([t_pts[:,:,:2], t_pts[:,:,102:], t_pts[:,:, 2:102]], dim = 2)
        dec_feat = torch.cat([c_pts, t_pts[:, :, :102]], dim = 0)
        dec_feat = torch.cat([torch.tile(action_token, (dec_feat.shape[0], 1, 1)), torch.tile(z, (dec_feat.shape[0], 1, 1)), dec_feat], dim = 2).permute(1,0,2)
        
        #dec_feat = torch.cat([torch.tile(z, (1, t_pts.shape[0], 1)), x_t.permute(1,0,2), y_t-y_t_pred], dim = 2) # 1 x n_pts x feat_dim
        dec_feat = self.encode_q(dec_feat) # 1 x n_tgt x feat_dim
        dec_feat = torch.mean(dec_feat, dim = 1).reshape(-1,1) #average across target points
        q_pred = dec_feat
        #q_pred = nn.Softmax(dim = 0)(q_pred)
        #q_pred = (q_pred - torch.min(q_pred))/(torch.max(q_pred) - torch.min(q_pred))
        q_pred = nn.Softmax(dim = 0)(dec_feat)
        return q_pred.reshape(-1,1)



