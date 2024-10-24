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
import argparse

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from models import Net
from diffusers.utils import check_min_version
logger = get_logger(__name__, log_level="INFO")



warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)