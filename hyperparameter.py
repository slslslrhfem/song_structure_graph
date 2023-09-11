class Hyperparameter:

    """

    Parameter for preprocessing, contains wasserstein VAE training. 

    """ 
    inst_num = 17


    batch_size= 64
    epochs = 100
    lr = 0.0001
    dim_h = 64
    n_z = 32
    LAMBDA = 1# It used in Wasserstein VAE, it depends on CONLON image normalizing method. 
    gan_LAMBDA = 1 # It used in Wasserstein VAE, in GAN based penalty.
    n_channel = inst_num * 2 # 2 for pattern AE, 34 for CBIR AE 
    sigma = 1
    pattern_length=4
    normalize_factor = 6.45873774657 # CONLON velocity channel normalizing factor.

    """

    Parameter for AE

    """

    AE_batch_size = 32
    AE_num_training_updates = 15000
    AE_num_hiddens = 128
    AE_num_residual_hiddens = 32
    AE_num_residual_layers = 2
    AE_embedding_dim = 512
    AE_learning_rate = 1e-3

    """
    
    Parameter for Graph Embedding Models

    """

    GEM_batch_size = 64
    GEM_learning_rate = 5e-3
    GEM_input_dim = 256 + AE_embedding_dim