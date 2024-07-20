from yacs.config import CfgNode
from logzero import logger

_defaults = CfgNode()
_defaults.seed = 42
_defaults.device = 'cpu'  # _defaults.network.device = 'cpu'
_defaults.backup = ['tfgat.py', 'datasets/vehicle.py', 'utils/scheduler.py',
                    'components/attention.py', 'components/encoder.py', 'components/decoder.py',
                    'components/feedforward.py', 'components/utils.py', 'components/DynamicGCf.py',
                    'components/embedding.py', 'components/non_ar_qgb.py', 'components/positional.py'
                    ]
_defaults.workspace = 'outputs/baseline'
_defaults.deterministic = False
_defaults.benchmark = True

_defaults.dataset = CfgNode()
_defaults.dataset.name = 'ngsim'
_defaults.dataset.train = '../data/origin_t_v_t/TestSet.mat'
_defaults.dataset.valid = '../data/origin_t_v_t/TestSet.mat'
_defaults.dataset.test = '../data/origin_t_v_t/TestSet.mat'
_defaults.dataset.downsample_rate = 2

# _defaults.dataset.name = 'highd'
# _defaults.dataset.train = '../data/highD/ValSet_.mat'
# _defaults.dataset.valid = '../data/highD/ValSet_.mat'
# _defaults.dataset.test = '../data/highD/ValSet_.mat'
# _defaults.dataset.downsample_rate = 5

_defaults.network = CfgNode()
_defaults.network.device = 'cpu'
_defaults.network.num_agents = 20
_defaults.network.max_len_edge = 90
_defaults.network.num_spatial_heads = 4
_defaults.network.num_spatial_encoders = 1
_defaults.network.num_temporal_heads = 4
_defaults.network.num_temporal_encoders = 1
_defaults.network.num_decoder_heads = 4
_defaults.network.num_decoders = 1
_defaults.network.attention_dims = 64
_defaults.network.feedforward_dims = 256
_defaults.network.attention_dropout = 0.2
_defaults.network.input_dropout = 0.
_defaults.network.feature_dropout = 0.
_defaults.network.maneuver = True
_defaults.network.multi_agents = False
_defaults.network.nast_random = 0.3
_defaults.network.num_spatial_time = 2
_defaults.network.use_nll = True  # _defaults.training.use_nll = True
_defaults.network.mult_traj = True#False  # only true in draw trajectory 多模态
_defaults.network.use_true_man = False
_defaults.network.use_hard_man = False  # only true when use_true_man is true
_defaults.network.hyper_scales = [5,9]

_defaults.training = CfgNode()
_defaults.training.pre_epochs = 4
_defaults.training.batch_size = 128
_defaults.training.num_workers = 8
_defaults.training.epochs = 8
_defaults.training.end_epochs = 11#9
_defaults.training.warmup_steps = 0
_defaults.training.warmup_factor = 0.5
_defaults.training.warmup_method = 'linear'
_defaults.training.standup_steps = 5
_defaults.training.scheduled_by_steps = False
_defaults.training.decay_method = 'None'
_defaults.training.decay_kwargs = []
_defaults.training.alpha = []
_defaults.training.beta = 1.
_defaults.training.gamma = 2.

_defaults.optimizer = CfgNode()
_defaults.optimizer.base_lr = 1e-3
_defaults.optimizer.base_weight_decay = 5e-5
_defaults.optimizer.bias_lr_factor = 1.
_defaults.optimizer.bias_decay_factor = 1.
_defaults.optimizer.name = 'Adam'
_defaults.optimizer.kwargs = []

_defaults.evaluation = CfgNode()
_defaults.evaluation.batch_size = 256
_defaults.evaluation.num_workers = 0
_defaults.evaluation.epoch = 10
_defaults.evaluation.path = ''
_defaults.evaluation.multimodal = False
_defaults.evaluation.softmaneuver = False
_defaults.evaluation.draw = False


def _check_configs(cfg):
    if not cfg.network.maneuver:
        if cfg.evaluation.multimodal:
            logger.warn("This model cannot enable multimodal in evaluation")
            cfg.evaluation.multimodal = False
        if cfg.evaluation.softmaneuver:
            logger.warn("This model cannot enable softmaneuver in evaluation")
            cfg.evaluation.softmaneuver = False

    if cfg.dataset.name.lower() == 'eth+ucy':
        if cfg.network.maneuver or (not cfg.network.multi_agents):
            logger.warn("This model cannot use maneuvers, or single track estimation")
            cfg.network.maneuver = False
            cfg.network.multi_agents = True


def get_default_configs():
    '''Return a default configuration for implemented TFGAT model'''
    cfg = _defaults.clone()
    _check_configs(cfg)
    cfg.freeze()
    return cfg


def get_modified_configs(config_file, command_list):
    '''Merge configuration from files and console lines into default configuration'''
    cfg = _defaults.clone()
    if config_file != "":
        cfg.merge_from_file(config_file)
    cfg.merge_from_list(command_list)
    _check_configs(cfg)
    cfg.freeze()
    return cfg
