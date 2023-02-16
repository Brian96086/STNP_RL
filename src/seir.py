import numpy as np
from numpy.random import binomial


def seir (num_days, beta_epsilon_flatten, num_simulations, cfg):
    mu = 1 #0.4
    all_cmpts = ['S', 'E', 'I', 'R', 'F']
    all_cases = ['E', 'I', 'R', 'F']

    x = range(num_days)

    ## model parameters
    ## initialization of california
    N = cfg.SIMULATOR.population #39512000

    ## save the number of death (mean, std) for each senario
    train_mean_list = []
    train_std_list = []
    train_list = []

    for i in range (len(beta_epsilon_flatten)):
        init_I = int(cfg.SIMULATOR.init_infected) #2000
        init_R = int(0) #0
        init_E = int(cfg.SIMULATOR.init_exposed) #2000

        ## save the number of individuals in each cmpt everyday
        dic_cmpts = dict()
        for cmpt in all_cmpts:
            dic_cmpts[cmpt] = np.zeros((num_simulations, num_days)).astype(int)

        dic_cmpts['S'][:, 0] = N - init_I - init_R - init_E
        dic_cmpts['I'][:, 0] = init_I
        dic_cmpts['E'][:, 0] = init_E
        dic_cmpts['R'][:, 0] = init_R
        

        ## save the number of new individuals entering each cmpt everyday
        dic_cases = dict()
        for cmpt in all_cmpts[1:]:
            dic_cases[cmpt] = np.zeros((num_simulations, num_days))

        ## run simulations
        for simu_id in range(num_simulations):
            for t in range(num_days-1):
                ## SEIR: stochastic
                flow_S2E = binomial(dic_cmpts['S'][simu_id, t], beta_epsilon_flatten[i,0] * dic_cmpts['I'][simu_id, t] / N)
                flow_E2I = binomial(dic_cmpts['E'][simu_id, t], beta_epsilon_flatten[i,1])
                flow_I2R = binomial(dic_cmpts['I'][simu_id, t], mu)
#                 print(t,flow_R2F)
                dic_cmpts['S'][simu_id, t+1] = dic_cmpts['S'][simu_id, t] - flow_S2E
                dic_cmpts['E'][simu_id, t+1] = dic_cmpts['E'][simu_id, t] + flow_S2E - flow_E2I
                dic_cmpts['I'][simu_id, t+1] = dic_cmpts['I'][simu_id, t] + flow_E2I - flow_I2R
                dic_cmpts['R'][simu_id, t+1] = dic_cmpts['R'][simu_id, t] + flow_I2R
                # dic_cmpts['F'][simu_id, t+1] = dic_cmpts['F'][simu_id, t] + flow_R2F

            
                ## get new cases per day
                dic_cases['E'][simu_id, t+1] = flow_S2E # exposed
                dic_cases['I'][simu_id, t+1] = flow_E2I # infectious
                dic_cases['R'][simu_id, t+1] = flow_I2R # removed
                # dic_cases['F'][simu_id, t+1] = flow_R2F # death 
        
        # rescale_cares_E = dic_cmpts['E'][...,1:]/N
        rescale_cares_I = dic_cmpts['I'][...,1:]/N*100
        # rescale_cares_R = dic_cmpts['R'][...,1:]/N

        train_list.append(rescale_cares_I)
        train_mean_list.append(np.mean(rescale_cares_I,axis=0))
        train_std_list.append(np.std(rescale_cares_I,axis=0))        

    train_meanset = np.stack(train_mean_list,0)
    train_stdset = np.stack(train_std_list,0)
    train_set = np.stack(train_list,0)
    return train_set, train_meanset, train_stdset

def build_dataset(cfg):

    num_days = cfg.SIMULATOR.num_days
    num_simulations = cfg.SIMULATOR.num_simulations

    [b_low, b_high, b_step], [e_low, e_high, e_step] = cfg.SIMULATOR.train_param
    beta = np.repeat(np.expand_dims(np.linspace(b_low, b_high, b_step),1),e_step,1) #1.1, 4.0, 30
    epsilon = np.repeat(np.expand_dims(np.linspace(e_low, e_high, e_step),0),b_step,0) #0.25, 0.65, 9
    beta_epsilon = np.stack([beta,epsilon],-1)
    beta_epsilon_train = beta_epsilon.reshape(-1,2)

    [b_low, b_high, b_step], [e_low, e_high, e_step] = cfg.SIMULATOR.val_param
    beta = np.repeat(np.expand_dims(np.linspace(b_low, b_high, b_step),1),e_step,1) #1.14, 3.88, 5
    epsilon = np.repeat(np.expand_dims(np.linspace(e_low, e_high, e_step),0),b_step,0) #0.29, 0.59, 3
    beta_epsilon = np.stack([beta,epsilon],-1)
    beta_epsilon_val = beta_epsilon.reshape(-1,2)

    [b_low, b_high, b_step], [e_low, e_high, e_step] = cfg.SIMULATOR.test_param
    beta = np.repeat(np.expand_dims(np.linspace(b_low, b_high, b_step),1),e_step,1) #1.24, 3.98, 5
    epsilon = np.repeat(np.expand_dims(np.linspace(e_low, e_high, e_step),0),b_step,0) #0.31, 0.61, 3
    beta_epsilon = np.stack([beta,epsilon],-1)
    beta_epsilon_test = beta_epsilon.reshape(-1,2)

    beta_epsilon_all = beta_epsilon_train
    yall_set, yall_mean, yall_std = seir(num_days,beta_epsilon_all,num_simulations, cfg)
    y_all = yall_set.reshape(-1,num_days-1)
    x_all = np.repeat(beta_epsilon_all,num_simulations,axis =0)

    yval_set, yval_mean, yval_std = seir(num_days,beta_epsilon_val,num_simulations, cfg)
    y_val = yval_set.reshape(-1,num_days-1)
    x_val = np.repeat(beta_epsilon_val,num_simulations,axis =0)


    ytest_set, ytest_mean, ytest_std = seir(num_days,beta_epsilon_test,num_simulations, cfg)
    y_test = ytest_set.reshape(-1, num_days-1)
    x_test = np.repeat(beta_epsilon_test,num_simulations,axis =0)

    scenario_list = [beta_epsilon_all, beta_epsilon_train, beta_epsilon_val, beta_epsilon_test]
    training_data = [x_all, yall_set, y_all]
    validation_data = [x_val, yval_set, y_val]
    test_data = [x_test, ytest_set, y_test]
    output_list = scenario_list, training_data, validation_data, test_data
    return output_list
