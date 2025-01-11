import torch

class Config:
    lambda_recon = 100

    beta1 = 0.5
    beta2 = 0.99

    n_epochs = 125
    input_dim = 1
    real_dim = 2
    display_step = 1
    batch_size = 128
    #lr = 0.0002
    target_shape = 32
    beta = 0.1
    device = 'cuda'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    ##########################
    smoothing=0.9
    #l1_weight=0.99
    base_lr_gen=3e-4
    base_lr_disc=6e-5
    base_lr_classifier=6e-5
    lr_decay_steps=6e4
    lr_decay_rate = 0.1
    num_workers = 4