from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.sim_type = "naive_seir"
_C.SIMULATOR.population = 100000
_C.SIMULATOR.num_days = 101
_C.SIMULATOR.num_simulations = 30
_C.SIMULATOR.init_infected = 2000
_C.SIMULATOR.init_exposed = 2000
#beta list of train, val, test 
# stored as ((beta_low, beta_high, beta_step), (ep_low, ep_high, ep_step)
_C.SIMULATOR.train_param = [[1.1, 4.0, 30],[0.25, 0.65, 9]]
_C.SIMULATOR.val_param = [[1.14, 3.88, 5],[0.29, 0.59, 3]]
_C.SIMULATOR.test_param = [[1.24, 3.98, 5],[0.31, 0.61, 3]]
# -----------------------------------------------------------------------------
# FILES
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.exp = ""
_C.DIR.snapshot = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.backbone = "deepQ"
_C.MODEL.hidden_dim = 256
_C.MODEL.refiner_proj_dim = 256
_C.MODEL.matcher_proj_dim = 256
_C.MODEL.dynamic_proj_dim = 128
_C.MODEL.refiner_layers = 6
_C.MODEL.matcher_layers = 6
_C.MODEL.repeat_times = 1
# dim of counter
_C.MODEL.counter_dim = 256
# use pretrained model
_C.MODEL.pretrain = True
_C.MODEL.lr = 1e-3
_C.MODEL.r_dim = 8
_C.MODEL.z_dim = 8
_C.MODEL.x_dim = 2
_C.MODEL.y_dim = 100


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# restore training from a checkpoint
_C.TRAIN.resume = "checkpoint_000000"
_C.TRAIN.counting_loss = "l1loss"
_C.TRAIN.batch_size = 1
_C.TRAIN.episode = 100
_C.TRAIN.stnp_epoch = 20000
_C.TRAIN.train_iter = 8
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
_C.TRAIN.n_display = 500
_C.TRAIN.patience = 5000
# optimizer and learning rate
_C.TRAIN.optimizer = "AdamW"
_C.TRAIN.lr_backbone = 0.01
_C.TRAIN.lr = 0.01
# milestone
_C.TRAIN.lr_drop = 200
# momentum
_C.TRAIN.momentum = 0.95
# weights regularizer
_C.TRAIN.weight_decay = 5e-4
# gradient clipping max norm
_C.TRAIN.clip_max_norm = 0.1
# number of data loading workers
_C.TRAIN.num_workers = 0
# frequency to display
_C.TRAIN.disp_iter = 20
_C.TRAIN.start_epoch = 0
_C.TRAIN.device = 'cpu'

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# the checkpoint to evaluate on
_C.VAL.resume = "model_best.pth.tar"
# currently only supports 1
_C.VAL.batch_size = 1
# frequency to display
_C.VAL.disp_iter = 10
# frequency to validate
_C.VAL.val_epoch = 10
# evaluate_only
_C.VAL.evaluate_only = False
_C.VAL.visualization = False