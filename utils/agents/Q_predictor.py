from torch import nn
import torch
from einops import rearrange

class Q_predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #given x_c, y_c, z, we find the representation for STNP p
        self.encode_p = nn.Sequential(
            nn.Linear(110, 512), #2+100 + 8(z_dim)
            nn.LayerNorm(512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )

        #given x_t, y_t, y_t_hat, z, we find the representation for STNP q
        self.encode_q = nn.Sequential(
            nn.Linear(210, 512), # 2+100*2 + 8(z_dim)
            nn.LayerNorm(512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        self.cat_linear = nn.Sequential(
            nn.Linear(120,64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),

            nn.Linear(32, 1),
        )
        
        self.q_conv1 = nn.Conv2d(out_channels = 1, in_channels = 256, kernel_size = 1)
        self.p_conv1 = nn.Conv2d(out_channels = 1, in_channels = 256, kernel_size = 1)

        #aggregate q and p features
        self.agg_qp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
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
        c_pts = ct_arr.reshape(1, ct_n, ct_dim)
        t_pts = tgt_arr.reshape(1, tgt_n, tgt_dim)
        z = z_arr.reshape(1, z_length)
        
#         c_pts = x["context_pts"]
#         z = x["latent_variable"]
#         t_pts = x["target_pts"]
        #import ipdb;ipdb.set_trace()
        #x_c,y_c  = torch.split(c_pts, [2, 100], dim = 1)
        #x_t, y_t, y_pred = torch.split(t_pts, [2,100,100], dim = 1)
        z = z.unsqueeze(1) # n_iter x 1 x z_dim (1 z per context pt set)
        p_feat = torch.cat([c_pts, torch.tile(z, (1, c_pts.shape[1], 1))], dim = 2)
        p_rep = self.encode_p(p_feat)

        q_feat = torch.cat([t_pts, torch.tile(z, (1, t_pts.shape[1], 1))], dim = 2)
        q_rep = self.encode_q(q_feat)

        cat_feat = torch.cat([p_rep, q_rep], dim = 1) #n_iter x (x_c+x_t) x 256
        cat_feat = rearrange(cat_feat, 'i n c -> i c n') #perform linear on (x_c + x_t) 

        #interact & engineer xc and xt information
        cat_feat = self.cat_linear(cat_feat)  #n x 256 x 1 -> x_c+xt information is aggregated
        cat_feat = torch.mean(cat_feat, dim = 0).reshape(1, -1) # 256 x 1

        #TODO: check dimension first
        #q_pred outputs the final q-values
        q_pred = self.agg_qp(cat_feat)

        return q_pred.reshape(-1,1)



