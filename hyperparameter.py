class Hyperparameter:

    """

    Parameter for preprocessing, contains wasserstein VAE training. 

    """


    batch_size= 64
    epochs = 100
    lr = 0.0001
    dim_h = 64
    n_z = 32
    LAMBDA = 1# It used in Wasserstein VAE, it depends on CONLON image normalizing method. 
    gan_LAMBDA = 1 # It used in Wasserstein VAE, in GAN based penalty.
    n_channel = 34 # 2 for pattern AE, 34 for CBIR AE 
    sigma = 1
    pattern_length=4
    normalize_factor = 6.45873774657 # CONLON velocity channel normalizing factor. velocity sum / 6.458.. = duration sum for our processing.

    """

    Parameter for VQ-VAE 

    """
    vq_batch_size = 32
    vq_num_training_updates = 15000
    vq_num_hiddens = 128
    vq_num_residual_hiddens = 32
    vq_num_residual_layers = 2
    vq_embedding_dim = 64
    vq_num_embeddings = 512
    vq_commitment_cost = 0.25
    vq_decay = 0.99
    vq_learning_rate = 1e-3

    """
    
    Parameter for Graph Embedding Models

    """

    GEM_batch_size = 64
    GEM_learning_rate = 1e-3
    GEM_input_dim = 512