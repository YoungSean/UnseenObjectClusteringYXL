import sys
import os
#print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Mask2Former'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
#print(os.path.join(os.path.dirname(__file__), '..', '..'))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.utils.visualizer import Visualizer
# from Mask2Former.mask2former import add_maskformer2_config
from mask2former import add_maskformer2_config
from datasets import OCIDDataset
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys, os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation, _vis_features, _vis_minibatch_segmentation_final
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.mean_shift import mean_shift_smart_init
from utils.evaluation import multilabel_metrics
import utils.mask as util_
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from torch.utils.data import DataLoader
# ignore some warnings
import warnings
warnings.simplefilter("ignore", UserWarning)


# build model
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
#cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
cfg_file = "../../Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
cfg.merge_from_file(cfg_file)
add_tabletop_config(cfg)
cfg.SOLVER.IMS_PER_BATCH = 1
# cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/output_RGB/model_0004999.pth"

# arguments frequently tuned
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
use_depth = True
weight_dir = "../../Mask2Former/"
weight_path = weight_dir + "depth_output_n1_lr5/model_0007999_n1_lr5.pth"
#weight_path = "../../Mask2Former/output_RGB_n2/model_final.pth"

if use_depth:
    cfg.INPUT.INPUT_IMAGE = 'DEPTH'
# test_dataset(dataset, predictor)


dataset = TableTopDataset(data_mapper=True,eval=True)
ocid_dataset = OCIDDataset(image_set="test")

use_my_dataset = True
for d in ["train", "test"]:
    if use_my_dataset:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
    else:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
    if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1:
        MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['object'])
    else:
        MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['background', 'object'])
metadata = MetadataCatalog.get("tabletop_object_train")

from topk_test_utils import Predictor_RGBD, test_dataset, test_sample



cfg.MODEL.WEIGHTS = weight_path
predictor = Predictor_RGBD(cfg)
test_sample(cfg, ocid_dataset[4], predictor, visualization=True)
# test_dataset(ocid_dataset, predictor, confident_score=0.9)
#test_dataset(ocid_dataset, predictor)
#test_dataset(cfg, dataset, predictor)