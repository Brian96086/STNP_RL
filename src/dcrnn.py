import torch
import torch.nn as nn

class DCRNNModel(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, device, init_func=torch.nn.init.normal_):
        super().__init__()
        self.repr_encoder = REncoder(x_dim+y_dim, r_dim) # (x,y)->r
        self.z_encoder = ZEncoder(r_dim, z_dim) # r-> mu, logvar
        self.decoder = Decoder(x_dim+z_dim, y_dim) # (x*, z) -> y*
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.device = device
    
    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        xy = torch.cat([x,y], dim=1)
        rs = self.repr_encoder(xy)
        r_agg = rs.mean(dim=0) # Average over samples
        return self.z_encoder(r_agg) # Get mean and variance for q(z|...)
    
    def sample_z(self, mu, logvar,n=1):
        """Reparameterisation trick."""
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(self.zdim).normal_()).to(self.device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n,self.zdim).normal_()).to(self.device)
        
        # std = torch.exp(0.5 * logvar)
        std = 0.1+ 0.9*torch.sigmoid(logvar)
        return mu + std * eps

    def KLD_gaussian(self):
        """Analytical KLD between 2 Gaussians."""
        mu_q, logvar_q, mu_p, logvar_p = self.z_mu_all, self.z_logvar_all, self.z_mu_context, self.z_logvar_context

        std_q = 0.1+ 0.9*torch.sigmoid(logvar_q)
        std_p = 0.1+ 0.9*torch.sigmoid(logvar_p)
        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        return torch.distributions.kl_divergence(p, q).sum()
        

    def forward(self, x_t, x_c, y_c, x_ct, y_ct):
        """
        """
        
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x_ct, y_ct)
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, self.zs)

#reference: https://chrisorm.github.io/NGP.html
class REncoder(torch.nn.Module):
    """Encodes inputs of the form (x_i,y_i) into representations, r_i."""
    
    def __init__(self, in_dim, out_dim, init_func = torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 16 #16
        self.l2_size = 8 #8
        
        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, out_dim)
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
        
    def forward(self, inputs):
        return self.l3(self.a2(self.l2(self.a1(self.l1(inputs)))))

class ZEncoder(torch.nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.logvar1_size = out_dim
        
        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.logvar1 = torch.nn.Linear(in_dim, self.m1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.logvar1.weight)
        
    def forward(self, inputs):
        

        return self.m1(inputs), self.logvar1(inputs)
    
class Decoder(torch.nn.Module):
    """
    Takes the x star points, along with a 'function encoding', z, and makes predictions.
    """
    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(Decoder, self).__init__()
        self.l1_size = 8 #8
        self.l2_size = 16 #16
        
        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, out_dim)
        
        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)
        
        self.a1 = torch.nn.Sigmoid()
        self.a2 = torch.nn.Sigmoid()
        
    def forward(self, x_pred, z):
        """x_pred: No. of data points, by x_dim
        z: No. of samples, by z_dim
        """
        zs_reshaped = z.unsqueeze(-1).expand(z.shape[0], x_pred.shape[0]).transpose(0,1)
        xpred_reshaped = x_pred
        
        xz = torch.cat([xpred_reshaped, zs_reshaped], dim=1)

        return self.l3(self.a2(self.l2(self.a1(self.l1(xz))))).squeeze(-1)