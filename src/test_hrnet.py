from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import sys
sys.path.append('/home/wanghongwei/WorkSpace/source/tools/pytorchviz/')

import torch
import torch.utils.data
from trains.train_factory import train_factory
from datasets.dataset_factory import get_dataset
from logger import Logger
from models.data_parallel import DataParallel
from models.model import create_model, load_model, save_model
from opts import opts


import os



def test_hrnet():
    arch = 'hrnet'
    heads = {'hm': 80, 'wh': 2, 'reg': 2}
    head_conv = 64  # 64, 128, 256
    cfg = '/home/wanghongwei/WorkSpace/source/detect/CenterNet/src/lib/config/w32_256x192_adam_lr1e-3.yaml'
    image = torch.randn(3, 3, 256, 192)
    model = create_model(arch=arch, heads=heads, head_conv=head_conv, cfg=cfg)
    result = model(image)
    return result

if __name__ == "__main__":
    res = test_hrnet()
    import pdb
    pdb.set_trace()
    print(res)
