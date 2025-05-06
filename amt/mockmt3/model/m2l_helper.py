m2l_hparams = {
    # GENERAL
    "mixed_precision": True,  # use mixed precision
    "seed": 42,  # seed for Pytorch

    # DATA
    "data_channels": 2,  # channels of input data
    "data_length": 64,  # sequence length of input data
    "data_length_test": 1024 // 4,  # sequence length of data used for testing
    "sample_rate": 44100,  # sampling rate of input/output audio
    "hop": 128 * 4,  # hop size of transformation
    "alpha_rescale": 0.65,
    "beta_rescale": 0.34,
    "sigma_data": 0.5,

    # MODEL
    "base_channels": 64,  # base channel number for architecture
    "layers_list": [2, 2, 2, 2, 2],  # number of blocks per each resolution level
    "multipliers_list": [1, 2, 4, 4, 4],  # base channels multipliers for each resolution level
    "attention_list": [0, 0, 1, 1, 1],  # attention configuration per resolution
    "freq_downsample_list": [1, 0, 0, 0],  # downsampling configuration per resolution
    "layers_list_encoder": [1, 1, 1, 1, 1],  # number of blocks per each resolution level (encoder)
    "attention_list_encoder": [0, 0, 1, 1, 1],  # attention configuration per resolution (encoder)
    "bottleneck_base_channels": 512,  # base channels for bottleneck
    "num_bottleneck_layers": 4,  # number of blocks before/after bottleneck
    "frequency_scaling": True,
    "heads": 4,  # number of attention heads
    "cond_channels": 256,  # dimension of time embedding
    "use_fourier": False,  # use random Fourier embedding
    "fourier_scale": 0.2,  # scale parameter for Fourier layer
    "normalization": True,  # use group normalization
    "dropout_rate": 0.,  # dropout rate
    "min_res_dropout": 16,  # apply dropout on equal/smaller feature maps
    "init_as_zero": True,  # initialize conv kernels before skip connections with zeros
    "bottleneck_channels": 32 * 2,  # encoder bottleneck channels
    "pre_normalize_2d_to_1d": True,  # pre-normalize 2D to 1D connection in encoder
    "pre_normalize_downsampling_encoder": True,  # pre-normalize downsampling in encoder
    "sigma_min": 0.002,  # minimum sigma
    "sigma_max": 80.,  # maximum sigma
    "rho": 7.  # rho parameter
}
