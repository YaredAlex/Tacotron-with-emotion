
class Hyperparams:

    """
    Hyperparameters
    """
    ##### Whether to restore the most recent model
    restore = False
    # pipeline
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence
    # data
    source = "EDS"
    data = "0015"
    max_duration = 10.0
    # signal processing
    sr = 22050 # Sample rate.
    n_fft = 1024 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = 256 # samples.
    win_length = 1024 # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    encoder_conv_channels = 256
    encoder_prenet_size = encoder_conv_channels 
    encoder_kernel_size = 5
    decoder_num_banks = 8
    decoder_conv_channels = 256
    decoder_kernel_size = 5
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5
    # Attention related parameters
    normalize_attention = False
    use_monotonic = False
    # training scheme
    lr = 0.001 # Initial learning rate.
    log_dir = "log"
    model_dir = "model_saved"
    batch_size = 16
    batches_per_group = 32
    # parameters used when preprocessing Korean data
    min_tokens = 30  # originally 50 30 is good for korean; set the mininum length of Korean text to be used for training
    min_n_frame = 30*r  # min_n_frame = reduction_factor * min_iters
    max_n_frame = 200*r
    frame_shift_ms=None # hop_size=  sample_rate *  frame_shift_ms / 1000
