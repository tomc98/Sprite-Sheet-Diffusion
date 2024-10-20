# ref: https://github.com/Zejun-Yang/AniPortrait
# Thanks to Huawei Wei and Zejun Yang and Zhisheng Wang
import warnings
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs

from diffusers.utils import check_min_version
logger = get_logger(__name__, log_level="INFO")



warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
