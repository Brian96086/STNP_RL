from torch import nn
import numpy as np
import torch
from einops import rearrange

class Q_predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
#         #given x_c, y_c, z, we find the representation for STNP p
#         self.conv_gt = nn.Conv1d(1, 128, 3, stride=1, padding = 1)
#         self.compress_gt = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64,1),
#         )
#         self.conv_ypred = nn.Conv1d(1, 128, 3, stride=1, padding = 1)
#         self.compress_ypred = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64,1),
#         )
#         self.encode_p = nn.Sequential(
#             nn.Linear(110, 256), #2+100 + 8(z_dim)
#             #nn.LayerNorm(256),
#             nn.ReLU(),
            
#             nn.Linear(256, 384),
#             #nn.LayerNorm(384),
#             nn.ReLU(),
            
#             nn.Linear(384, 512),
#             #nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, 512),
#             #nn.LayerNorm(512),
#             nn.ReLU(),

# #             nn.Linear(512, 256),
# #             nn.LayerNorm(256),
# #             nn.LeakyReLU(),
#         )

#         #given x_t, y_t, y_t_hat, z, we find the representation for STNP q
#         self.encode_q = nn.Sequential(
#             nn.Linear(110, 256), # 2+100*2 + 8(z_dim)
#             #nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(256, 384),
#             #nn.LayerNorm(384),
#             nn.ReLU(),
            
#             nn.Linear(384, 512),
#             #nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, 512),
#             #nn.LayerNorm(512),
#             nn.ReLU(),

#         )
#         self.cat_linear = nn.Sequential(
#             nn.Linear(120,64),
#             #nn.LayerNorm(64),
#             nn.ReLU(),

#             nn.Linear(64, 32),
#             #nn.LayerNorm(32),
#             nn.ReLU(),

#             nn.Linear(32, 1),
#         )
        
#         self.q_conv1 = nn.Conv2d(out_channels = 1, in_channels = 256, kernel_size = 1)
#         self.p_conv1 = nn.Conv2d(out_channels = 1, in_channels = 256, kernel_size = 1)

#         #aggregate q and p features
#         self.agg_qp = nn.Sequential(
# #             nn.Linear(512, 256),
# #             nn.LayerNorm(256),
# #             nn.Linear(256, 270)
#             nn.Linear(512, 384),
#             #nn.LayerNorm(384),
#             nn.ReLU(),
#             nn.Linear(384, 270),
#             nn.ReLU(),
#             nn.Linear(270, 270),

#         )

        self.z_conv = nn.Sequential(
#             nn.Linear(8, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 270)
            nn.Linear(110, 192),
            nn.LeakyReLU(),
            nn.Linear(192, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 270)
        )
    
    def forward(self, x):
         #TODO 1: add convolution on y_t for both encode_p and encode_q
        #TODO 2: after q_rep, p_rep, let model learn the relationship based on similar z value
        #TODO: check when batch changes
        x = x.reshape(-1,1)
        
        state_arr, ct_shape, tgt_shape  = torch.split(x, [x.shape[0]-4, 2, 2], dim = 0)
        ct_n, ct_dim = int(ct_shape[0]), int(ct_shape[1]) #context
        tgt_n, tgt_dim = int(tgt_shape[0]), int(tgt_shape[1]) #target
        z_length  = state_arr.shape[0]-ct_n*ct_dim-tgt_n*tgt_dim
        ct_arr, tgt_arr, z_arr = torch.split(state_arr.unsqueeze(0), [ct_n*ct_dim, tgt_n*tgt_dim, z_length], dim = 1)
        c_pts = ct_arr.reshape(ct_n, 1, ct_dim)
        t_pts = tgt_arr.reshape(tgt_n, 1, tgt_dim)
        z = z_arr.reshape(1, z_length)
        
#         #import ipdb;ipdb.set_trace()
#         x_c, y_c = torch.split(c_pts, [2,100], dim =2)
#         x_t, y_t, y_t_pred = torch.split(t_pts, [2, 100,100], dim = 2) 
#         y_c = self.conv_gt(y_c)
#         y_t, y_t_pred = self.conv_gt(y_t), self.conv_ypred(y_t_pred) #use three convolutions, or try using 1 conv on both p,q
#         y_c, y_t, y_t_pred = self.compress_gt(y_c.permute(0, 2, 1)), self.compress_gt(y_t.permute(0, 2, 1)), self.compress_ypred(y_t_pred.permute(0, 2, 1)) # n_pts x feat_dim x 1
#         c_pts = torch.cat([x_c.permute(1,0,2), y_c.permute(2, 0, 1)], dim = 2) # 1 x n_pts x feat_dim
#         t_pts = torch.cat([x_t.permute(1,0,2), y_t.permute(2, 0, 1)-y_t_pred.permute(2, 0, 1)], dim = 2) # 1 x n_pts x feat_dim
        

#         z = z.unsqueeze(1) # n_iter x 1 x z_dim (1 z per context pt set)
#         p_feat = torch.cat([c_pts, torch.tile(z, (1, c_pts.shape[1], 1))], dim = 2)
#         p_rep = self.encode_p(p_feat)

#         q_feat = torch.cat([t_pts, torch.tile(z, (1, t_pts.shape[1], 1))], dim = 2)
#         q_rep = self.encode_q(q_feat)

#         cat_feat = torch.cat([p_rep, q_rep], dim = 1) #n_iter x (x_c+x_t) x 256
#         cat_feat = rearrange(cat_feat, 'i n c -> i c n') #perform linear on (x_c + x_t) 

#         #interact & engineer xc and xt information
#         cat_feat = self.cat_linear(cat_feat)  #n x 256 x 1 -> x_c+xt information is aggregated
#         cat_feat = torch.mean(cat_feat, dim = 0).reshape(1, -1) # 256 x 1

#         #TODO: check dimension first
#         #q_pred outputs the final q-values
#         q_pred = self.agg_qp(cat_feat)
#         q_pred = nn.Softmax()(q_pred*0.1)
        q_pred = self.z_conv(torch.cat([c_pts.permute(1,0,2), torch.tile(z, (1, c_pts.shape[0], 1))], dim = 2))
        q_pred = torch.mean(q_pred, dim = 1).reshape(-1,1)
        #print('q_pred stats = {}'.format(torch.quantile(q_pred, q = torch.tensor([0.1, 0.3, 0.5,0.7, 0.9]))))
        #q_pred = nn.Softmax(dim = 1)(q_pred)
        q_pred = nn.Sigmoid()(q_pred)
        

        return q_pred.reshape(-1,1)



