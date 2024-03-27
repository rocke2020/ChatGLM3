import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm
from torch.distributed import init_process_group


logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')

def ddp_setup_torchrun():
    init_process_group(backend="nccl")


def main():
    pass


if __name__ == '__main__':
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    main()
