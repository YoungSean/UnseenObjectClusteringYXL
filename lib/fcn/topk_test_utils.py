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
# cfg = get_cfg()
# add_deeplab_config(cfg)
# add_maskformer2_config(cfg)
# #cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
# cfg_file = "../../Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
# cfg.merge_from_file(cfg_file)
# add_tabletop_config(cfg)
# # cfg.INPUT.INPUT_IMAGE = 'DEPTH'
# cfg.SOLVER.IMS_PER_BATCH = 1
# # cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/output_RGB/model_0004999.pth"
# cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1


# dataset = TableTopDataset(data_mapper=True,eval=True)
# ocid_dataset = OCIDDataset(image_set="test")
#
# use_my_dataset = True
# #DatasetCatalog.register("tabletop_object_train", getTabletopDataset)
# for d in ["train", "test"]:
#     if use_my_dataset:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
#     else:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
#     if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1:
#         MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['object'])
#     else:
#         MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['background', 'object'])
# metadata = MetadataCatalog.get("tabletop_object_train")

# Reference: https://www.reddit.com/r/computervision/comments/jb6b18/get_binary_mask_image_from_detectron2/

def get_confident_instances(outputs, topk=True, score=0.9, num_class=2):
    """
    Extract objects with high prediction scores.
    """
    instances = outputs["instances"]
    if topk:
        # we need to remove background predictions
        # keep only object class
        if num_class==2:
            return instances[instances.pred_classes == 1]
        else:
            return instances
    confident_instances = instances[instances.scores > score]
    return confident_instances

def combine_masks(instances):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
    """
    mask = instances.get('pred_masks').to('cpu').numpy()
    num, h, w = mask.shape
    bin_mask = np.zeros((h, w))
    num_instance = len(mask)
    # if there is not any instance, just return a mask full of 0s.
    if num_instance == 0:
        return bin_mask

    for m, object_label in zip(mask, range(2, 2+num_instance)):
        label_pos = np.nonzero(m)
        bin_mask[label_pos] = object_label
    # filename = './bin_masks/001.png'
    # cv2.imwrite(filename, bin_mask)
    return bin_mask

class Predictor_RGBD(DefaultPredictor):

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            height, width = 480, 640
            if self.cfg.INPUT.INPUT_IMAGE == "DEPTH":
                depth = original_image
                inputs = {"height": height, "width": width, "depth": depth}
            else:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions

def test_sample(cfg, sample, predictor, visualization = False, topk=True, confident_score=0.9):
    im = cv2.imread(sample["file_name"])
    if "label" in sample.keys():
        gt = sample["label"].squeeze().numpy()
    else:
        gt = sample["labels"].squeeze().numpy()

    if cfg.INPUT.INPUT_IMAGE == "DEPTH":
        outputs = predictor(sample["depth"])
    else:
        outputs = predictor(im)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score, num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    binary_mask = combine_masks(confident_instances)
    metrics = multilabel_metrics(binary_mask, gt)
    #print(f"metrics: ", metrics)
    ## Visualize the result
    if visualization:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image", visual_result)
        cv2.waitKey(0)
        # cv2.waitKey(100000)
        cv2.destroyAllWindows()
    return metrics

def test_dataset(cfg,dataset, predictor, visualization=False, topk=True, confident_score=0.9):
    metrics_all = []
    for i in trange(len(dataset)):
        metrics = test_sample(cfg, dataset[i], predictor, visualization=visualization, topk=topk, confident_score=confident_score)
        metrics_all.append(metrics)
    # for i in tqdm(dataset):
    #     metrics = test_sample(i, predictor, visualization=visualization)
    #     metrics_all.append(metrics)
    print('========================================================')
    if not topk:
        print("Mask threshold: ", confident_score)
    print("weight: ", cfg.MODEL.WEIGHTS)
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================END=================================')

def test_dataset_with_weight(weight_path, cfg, dataset,topk=True, confident_score=0.9):
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Predictor_RGBD(cfg)
    test_dataset(cfg, dataset, predictor, topk=topk, confident_score=confident_score)

def test_dataset_with_weight(weight_path, cfg, sample,topk=True, confident_score=0.9):
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Predictor_RGBD(cfg)
    test_dataset(cfg, sample, predictor, topk=topk, confident_score=confident_score)

#
# use_depth = True
# if use_depth:
#     cfg.INPUT.INPUT_IMAGE = 'DEPTH'
# # test_dataset(dataset, predictor)
#
# #weight_path = "../../Mask2Former/output_RGB_n2/model_final.pth"
# #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
# weight_path = "../../Mask2Former/depth_output_n1_lr5/model_final.pth"
# cfg.MODEL.WEIGHTS = weight_path
# predictor = Predictor_RGBD(cfg)
# test_sample(ocid_dataset[4], predictor, visualization=True)
# # test_dataset(ocid_dataset, predictor, confident_score=0.9)
# #test_dataset(ocid_dataset, predictor)
# test_dataset(dataset, predictor)
# print(ocid_dataset[4])


