# --------------------------------------------------------
# EdgeSAM Config
# Based on the code: TinyViT
#   (https://github.com/microsoft/Cream/tree/main/TinyViT)
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'sa1b'
# Dataset mean/std type
_C.DATA.MEAN_AND_STD_TYPE = "default"
# Input image size
_C.DATA.IMG_SIZE = 1024
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Data image filename format
_C.DATA.FNAME_FORMAT = '{}.jpeg'
# Data debug, when debug is True, only use few images
_C.DATA.DEBUG = False
# Filter the box instances by area
_C.DATA.FILTER_BY_AREA = None
# Load GT masks
_C.DATA.LOAD_GT_MASK = False
# Apply mask NMS with the given threshold
_C.DATA.MASK_NMS = 1.0
# Box jitter
_C.DATA.BOX_JITTER = False
# Number of samples, -1 means all
_C.DATA.NUM_SAMPLES = -1

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'rep_vit_m1'
# Model name (if None, use the config file name)
_C.MODEL.NAME = None
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000

# DISTILL
_C.DISTILL = CN()
_C.DISTILL.TEACHER_EMBED_PATH = ''
_C.DISTILL.SAVE_TEACHER_EMBED = False
_C.DISTILL.EMBED_DIM = 100
# No randomness during teacher embeddings saving.
_C.DISTILL.NO_RAND = False
# The downsampling stride of the SAM image encoder is 16, while that of most backbones is 32. To align the feature
# size, one way is to fuse features from the last two stages with FPN, the other way is to remove the downsampling in
# the last stage.
_C.DISTILL.FUSE = False
# Apply pixel-wise feature distillation. (If the value > 0, then enabled and the value represents the loss weight)
_C.DISTILL.PIXEL_WISE = -1.0
# Apply channel-wise feature distillation.
_C.DISTILL.CHANNEL_WISE = -1.0
# Apply pair-wise feature correlation distillation.
_C.DISTILL.CORRELATION = -1.0
# Apply distillation only on encoder.
_C.DISTILL.ENCODER_ONLY = True
# Distill on the image feature output from the mask decoder.
_C.DISTILL.DECODER_FEAT = -1.0
# Distill on the mask token (refined query) output from the mask decoder.
_C.DISTILL.DECODER_QUERY = -1.0
# Distill on the mask logits from the decoder (BCE Loss).
_C.DISTILL.DECODER_BCE = -1.0
# Distill on the mask logits from the decoder (Focal Loss).
_C.DISTILL.DECODER_FOCAL = -1.0
# Distill on the mask logits from the decoder (Dice Loss).
_C.DISTILL.DECODER_DICE = -1.0
# Distill on the IoU prediction from the decoder.
_C.DISTILL.DECODER_IOU = -1.0
# Distill on the attention map from the decoder.
_C.DISTILL.DECODER_ATTN = -1.0
# The number of max allowed prompts in a single image.
_C.DISTILL.MAX_ALLOWED_PROMPTS = 8
# Whether to freeze the image encoder of the student during training.
_C.DISTILL.FREEZE_IMAGE_ENCODER = False
# Whether to freeze the prompt encoder of the student during training.
_C.DISTILL.FREEZE_PROMPT_ENCODER = True
# Whether to freeze the mask decoder of the student during training.
_C.DISTILL.FREEZE_MASK_DECODER = False
# Infer the decoder for how many iterations.
_C.DISTILL.DECODE_ITERS = 1
# For each refine iteration, sample how many new point prompts.
_C.DISTILL.POINTS_PER_REFINE_ITER = 1
# Whether to apply iterative decoding for box prompts.
_C.DISTILL.ITER_ON_BOX = False
# Enale Point Rend sampling for logits distillation and/or task loss.
_C.DISTILL.POINT_REND_SAMPLING = False
# When performing logits distillation, use teacher logits or teacher labels (by thresholding).
_C.DISTILL.USE_TEACHER_LOGITS = False
# Initialize the weights of the prompt encoder and mask decoder of the student from teacher.
_C.DISTILL.INIT_FROM_TEACHER = True
# Temperature for mask prediction (prior to sigmoid).
_C.DISTILL.TEMPERATURE = 1.0
# Convert prompt box to prompt point (center point)
_C.DISTILL.PROMPT_BOX_TO_POINT = False
# Convert gt mask to prompt point (random point)
_C.DISTILL.PROMPT_MASK_TO_POINT = False
# Prompt types on the initial prompt (support box and point)
_C.DISTILL.PROMPT_TYPE = ['box']
# Apply Lora on the attention weights (query and value projection)
_C.DISTILL.LORA = False
# RPN head used during inference
_C.DISTILL.RPN_HEAD = 'none'
# number of masks for outputs (choices: 1, 3, 4)
_C.DISTILL.MULTIMASK_OUTPUT = 1
# If False, when using box as prompt, the MULTIMASK_OUTPUT will be set to 1
_C.DISTILL.MULTIMASK_ON_BOX = False
# The upsampling mode used in fuse_neck (CoreML doesn't support bicubic)
_C.DISTILL.UPSAMPLE_MODE = 'bicubic'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# train learning rate decay
_C.TRAIN.LAYER_LR_DECAY = 1.0
# batch norm is in evaluation mode when training
_C.TRAIN.EVAL_BN_WHEN_TRAINING = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.FIND_UNUSED_PARAMETERS = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.disable_amp or args.only_cpu:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    if args.local_rank is None and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    if config.MODEL.NAME is None:
        config.MODEL.NAME = os.path.basename(args.cfg).split('.')[0]

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args=None):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if args is not None:
        update_config(config, args)

    return config
