import torch 
import numpy as np
import matplotlib.pyplot as plt
from utils.trainer.Trainer import Trainer


device = torch.device("cpu")
large = 25; med = 19; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': 20,
          'figure.figsize': (27, 8),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': med}
plt.rcParams.update(params)

def test(dcrnn, x_train, y_train, x_test, z_dim):
    with torch.no_grad():
      z_mu, z_logvar = data_to_z_params(dcrnn, x_train.to(device),y_train.to(device))
      
      output_list = []
      for i in range (len(x_test)):
          zsamples = sample_z(z_mu, z_logvar, z_dim) 
          output = dcrnn.decoder(x_test[i:i+1].to(device), zsamples).cpu()
          output_list.append(output.detach().numpy())
    
    return np.concatenate(output_list)

def train(dcrnn, opt, cfg, n_epochs, x_train, y_train, x_val, y_val, x_test, y_test, n_display=500, patience = 5000, beta_epsilon_all = []): #7000, 1000
    train_losses = []
    # mae_losses = []
    # kld_losses = []
    val_losses = []
    test_losses = []

    means_test = []
    stds_test = []
    N = 100000 #population
    min_loss = 0. # for early stopping
    wait = 0
    min_loss = float('inf')
    c_arr, z_arr,t_arr= [], [], []
    val_arr, test_arr = [], []
    rewards = []
    
    for t in range(n_epochs): 
        opt.zero_grad()
        #Generate data and process
        x_context, y_context, x_target, y_target = random_split_context_target(
                                x_train, y_train, int(len(y_train)*0.1)) #0.25, 0.5, 0.05,0.015, 0.01
        # print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)    

        x_c = torch.from_numpy(x_context).float().to(device)
        x_t = torch.from_numpy(x_target).float().to(device)
        y_c = torch.from_numpy(y_context).float().to(device)
        y_t = torch.from_numpy(y_target).float().to(device)

        x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
        y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

        y_pred = dcrnn(x_t, x_c, y_c, x_ct, y_ct)

        train_loss = N * MAE(y_pred, y_t)/100 + dcrnn.KLD_gaussian()
        mae_loss = N * MAE(y_pred, y_t)/100
        kld_loss = dcrnn.KLD_gaussian()
        if(len(beta_epsilon_all) !=0): # case where we are pre-training in stage 0
            score_arr = calculate_score(cfg, dcrnn, x_train, y_train, beta_epsilon_all)
            rewards.append(torch.from_numpy(score_arr))
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(dcrnn.parameters(), 5) #10
        opt.step()
        
        #val loss
        y_val_pred = test(dcrnn, torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_val).float(), cfg.MODEL.z_dim)
        val_loss = N * MAE(torch.from_numpy(y_val_pred).float(),torch.from_numpy(y_val).float())/100
        #test loss
        y_test_pred = test(dcrnn, torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_test).float(), cfg.MODEL.z_dim)
        test_loss = N * MAE(torch.from_numpy(y_test_pred).float(),torch.from_numpy(y_test).float())/100
        
        c_arr.append(torch.cat([x_c, y_c], dim = 1)) #n_iter, n_xc, 102
        z_arr.append(dcrnn.z_mu_all)
        t_arr.append(torch.cat([x_t, y_t, y_pred], dim = 1))
        
        val_arr.append(torch.from_numpy(y_val_pred))
        test_arr.append(torch.from_numpy(y_test_pred))
        if t % n_display ==0:
            print('train loss:', train_loss.item(), 'mae:', mae_loss.item(), 'kld:', kld_loss.item())
            print('val loss:', val_loss.item(), 'test loss:', test_loss.item())

        if t % (n_display/10) ==0:
            print('curr_epoch = {}'.format(t))
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            test_losses.append(test_loss.item())
            # mae_losses.append(mae_loss.item())
            # kld_losses.append(kld_loss.item())

        #early stopping
        if val_loss < min_loss:
            wait = 0
            min_loss = val_loss
            
        elif val_loss >= min_loss:
            wait += 1
            if wait == patience:
                print('Early stopping at epoch: %d' % t)
                return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all

    c_arr = torch.stack(c_arr, dim = 0)
    z_arr = torch.stack(z_arr, dim = 0)
    t_arr = torch.stack(t_arr, dim = 0)
    val_arr = torch.stack(val_arr, dim = 0)
    test_arr = torch.stack(test_arr, dim = 0)
    reward_arr = torch.stack(rewards, dim = 0)
    data_scenarios = {"context_pts": c_arr, "latent_variable": z_arr,"target_pts": t_arr,
          "valid_pred": val_arr, "test_pred": test_arr, "valid_gt":y_val, "test_gt":y_test, "gt_rewards":reward_arr}
        
    return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all,data_scenarios


def MAE(pred, target):
    loss = torch.abs(pred-target)
    return loss.mean()

def random_split_context_target(x,y, n_context):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)

def sample_z(mu, logvar,z_dim, n=1):
    """Reparameterisation trick."""
    if n == 1:
        eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_())
    else:
        eps = torch.autograd.Variable(logvar.data.new(n,z_dim).normal_())
    
    std = 0.1+ 0.9*torch.sigmoid(logvar)
    return mu + std * eps

def data_to_z_params(dcrnn, x, y):
    """Helper to batch together some steps of the process."""
    xy = torch.cat([x,y], dim=1)
    rs = dcrnn.repr_encoder(xy)
    r_agg = rs.mean(dim=0) # Average over samples
    return dcrnn.z_encoder(r_agg) # Get mean and variance for q(z|...)


def select_data(cfg, x_train, y_train, beta_epsilon_all, yall_set, score_array, selected_mask):

    mask_score_array = score_array*(1-selected_mask)
    # print('mask_score_array',mask_score_array)
    select_index = np.argmax(mask_score_array)
    print('select_index:',select_index)


    selected_x = beta_epsilon_all[select_index:select_index+1]
    selected_y = yall_set[select_index]

    x_train1 = np.repeat(selected_x,cfg.SIMULATOR.num_simulations,axis =0)
    x_train = np.concatenate([x_train, x_train1],0)
    
    y_train1 = selected_y.reshape(-1,100)
    y_train = np.concatenate([y_train, y_train1],0)
 
    selected_mask[select_index] = 1
    
    return x_train, y_train, selected_mask


def calculate_score(cfg, dcrnn, x_train, y_train, beta_epsilon_all):
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    # query z_mu, z_var of the current training data
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(dcrnn, x_train.to(device),y_train.to(device), )

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
                output = dcrnn.decoder(x_search[j:j+1].to(device), zsamples).cpu()
                output_list.append(output.detach().numpy())

            y_search = np.concatenate(output_list)
            y_search = torch.from_numpy(y_search).float()

            x_search_all = torch.cat([x_train,x_search],dim=0)
            y_search_all = torch.cat([y_train,y_search],dim=0)

            # generate z_mu_search, z_var_search
            z_mu_search, z_logvar_search = data_to_z_params(dcrnn, x_search_all.to(device),y_search_all.to(device))
            
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

"""BO search:"""

def mae_plot(mae, selected_mask,i,j):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    selected_mask = selected_mask.reshape(30,9)
    mae_min, mae_max = 0, 1200

    fig, ax = plt.subplots(figsize=(16, 7))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, mae, cmap='binary', vmin=mae_min, vmax=mae_max)
    ax.set_title('MAE Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.savefig('mae_plot_seed%d_itr%d.pdf' % (i,j))

def score_plot(score, selected_mask,i,j):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    score_min, score_max = 0, 1
    selected_mask = selected_mask.reshape(30,9)
    score = score.reshape(30,9)
    fig, ax = plt.subplots(figsize=(16, 7))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, score, cmap='binary', vmin=score_min, vmax=score_max)
    ax.set_title('Score Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.savefig('score_plot_seed%d_itr%d.pdf' % (i,j))

def MAE_MX(y_pred, y_test):
    N = 100000
    y_pred = y_pred.reshape(30,9, 30, 100)*N/100
    y_test = y_test.reshape(30,9, 30, 100)*N/100
    mae_matrix = np.mean(np.abs(y_pred - y_test),axis=(2,3))
    mae = np.mean(np.abs(y_pred - y_test))
    return mae_matrix, mae


def train_DQN(dqn, dcrnn, beta_epsilon_all, scenarios, config, cfg, episodes):
    '''
    Premise
    - Within the STNP train(), we can retrieve lots of (x_ct, y_ct, x_t, y_t, z)
    - By training an STNP model for one epoch, we can retrieve a lot of points
    - Eg: retrieve 1000 data points and compute the reward

    Stage1: Given fixed latent variable distribution, learn the reward function and informative parameter 
    - state = [z, beta_epsilon_all, x_train, y_train]
    - reward = acquisition function, 
    - Model will output the best action -> correspond to parameter set
    - env.step() will be fixed (selected parameter will have have its q-value masked to min value)
      - or we can mask the chosen beta_epsilon to be 0 within the state vector
    - To encourage the model to select the parameter with the best reward, we perform exponential/reciprocal decay on rewards
    - 
    Stage2: Given the model's decent adaptability to learn mapping from states to reward given fixed latent variable z

    '''
    #context_pts = [x_c, y_c], x_c = (n_c,2), y_c = (n_c,100)
    #target_pts = [x_t, y_t, y_t_pred], x_t = (n_t, 2), y_t = y_t_pred = (n_t, 100)
    context_pts = scenarios["context_pts"]
    zs = scenarios["latent_variable"]
    target_pts = scenarios["target_pts"]
    valid_pred = scenarios["valid_pred"]
    test_pred = scenarios["test_pred"]

    #context_pts, zs, target_pts = scenarios 
    n_c = context_pts.shape[0]
    trainer = Trainer(config, cfg, [dqn])
    trainer.run_games_for_agents()

    #stack x_c and y_c -> (n_c, 102) (n_c data points, 102 features)
    #each z captures the representation of x_c, y_c -> stack with z (n_c, 103) 

    # input_feat = torch.cat([context_pts, z], dim = 1)
    # for t in range(episodes):
    #     q_values = dqn.forward(input_feat)
    #     q_values = q_values * mask
    #     max_ind = np.argmax(q_values)
    #     best_param = q_values[max_ind]
    #     mask[best_param] = 0
    #     reward = env.step(best_param)
        
    

    # (x_ct, y_ct, x_t, y_t, z)
    #
    
    return None
    