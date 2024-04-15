from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import re
import time
import datetime
import torch as th
import torchvision.models as models
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter
from src.utils.renderer import Renderer, visualize_reconstruction_and_att_local, visualize_reconstruction_no_text
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

class HandMesh():
    def __init__(self) -> None:
        self.num_gpus = 1
        self.seed = 88
        self.num_workers = 4
        self.device = th.device('cuda')
        # Setup CUDA, GPU
        set_seed(self.seed, self.num_gpus)
        os.environ['OMP_NUM_THREADS'] = str(self.num_workers)
        # Mesh and MANO utils
        self.mano_model = MANO().to(self.device)
        self.mano_model.layer = self.mano_model.layer.cuda()
        self.mesh_sampler = Mesh()

        # Renderer for visualization
        self.renderer = Renderer(faces=self.mano_model.face)