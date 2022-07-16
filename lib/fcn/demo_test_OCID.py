import sys
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.utils.visualizer import Visualizer
from Mask2Former.mask2former import add_maskformer2_config
from datasets import OCIDObject
print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
print(os.path.join(os.path.dirname(__file__), '..', '..'))
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
from Mask2Former.tabletop_config import add_tabletop_config
from torch.utils.data import DataLoader
# build model
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
#cfg_file = "configs/cityscapes/instance-segmentation/Base-Cityscapes-InstanceSegmentation.yaml"
cfg.merge_from_file(cfg_file)
add_tabletop_config(cfg)
cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/output/model_final.pth"
model = build_model(cfg)
model.eval()


dataset = TableTopDataset(data_mapper=True,eval=True)
ocid_dataset = OCIDObject(image_set="test")
# print(len(dataset))
#sample = dataset[3]
#gt = sample["label"].squeeze().numpy()
# with torch.no_grad():
#   prediction = model(sample)
#
# outputs = prediction[0]


use_my_dataset = True
#DatasetCatalog.register("tabletop_object_train", getTabletopDataset)
for d in ["train", "test"]:
    if use_my_dataset:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
    else:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
    MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['__background__', 'object'])
metadata = MetadataCatalog.get("tabletop_object_train")

# Reference: https://www.reddit.com/r/computervision/comments/jb6b18/get_binary_mask_image_from_detectron2/

def get_confident_instances(outputs, score=0.9):
    """
    Extract objects with high prediction scores.
    """
    instances = outputs["instances"]
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

img_path = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/rgb_00003.jpeg"
predictor = DefaultPredictor(cfg)
def test_sample(sample, predictor, visualization = False, confident_score=0.9):
    im = cv2.imread(sample["file_name"])
    gt = sample["label"].squeeze().numpy()
    print(gt)
    # im = cv2.imread("/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/ocid001.png")
    outputs = predictor(im)
    confident_instances = get_confident_instances(outputs, score=confident_score)
    binary_mask = combine_masks(confident_instances)
    metrics = multilabel_metrics(binary_mask, gt)
    print(f"metrics: ", metrics)
    ## Visualize the result
    if visualization:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        cv2.imshow("image", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return metrics

# visualizeResult(img_path, gt)

class ObjectEvaluator(DatasetEvaluator):

    def reset(self):
        self.metrics_all = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt = input["label"].squeeze().numpy()
            confident_instances = get_confident_instances(outputs, score=0.8)
            binary_mask = combine_masks(confident_instances)
            metrics = multilabel_metrics(binary_mask, gt)
            self.metrics_all.append(metrics)

    def evaluate(self):
        print('========================================================')
        result = {}
        num = len(self.metrics_all)
        print('%d images' % num)
        print('========================================================')
        for metrics in self.metrics_all:
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
        print('====================Refined=============================')

cfg.DATASETS.TEST = ("tabletop_object_train", )
cfg = cfg.clone()  # cfg can be modified by model
model = build_model(cfg)
model.eval()

if len(cfg.DATASETS.TEST):
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

# if use_my_dataset:
#     dataset = TableTopDataset(image_set="train", data_mapper=True, eval=True)#, data_mapper=DatasetMapper(cfg, is_train=True))
#     dataloader = DataLoader(dataset, batch_size=1)
def get_all_inputs_outputs(dataloader):
    for data in dataloader:
        filter_batch = []
        if data["annotations"]:
            filter_batch.append(data)
        if len(filter_batch) == 0:
            continue
        # sample = data
        yield filter_batch, model(filter_batch)

# eval_results = inference_on_dataset(
#     model,
#     dataloader,
#     ObjectEvaluator())
# evaluator = ObjectEvaluator()
# evaluator.reset()
# for inputs, outputs in get_all_inputs_outputs():
#   evaluator.process(inputs, outputs)
# eval_results = evaluator.evaluate().


def test_dataset(dataset, predictor):
    metrics_all = []
    for i in range(len(dataset)):
        metrics = test_sample(dataset[i], predictor)
        metrics_all.append(metrics)
    print('========================================================')
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

# test_dataset(dataset, predictor)
# test_sample(dataset[0], predictor, visualization=True)

# test_sample(ocid_dataset[0], predictor, visualization=True)
print(ocid_dataset[0]["file_name"])
#print(cv2.imread(ocid_dataset[0]["file_name"]))
#print(cv2.imread("/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/ocid001.png"))
