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
cfg.SOLVER.IMS_PER_BATCH = 1 #
# cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/output_RGB/model_0004999.pth"

# arguments frequently tuned
cfg.TEST.DETECTIONS_PER_IMAGE = 20
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
use_depth = True
weight_path = "../../Mask2Former/depth_R50_lr4_noflip2/model_final.pth"  # depth_lr4_wo_pretrained/model_final.pth #depth_output_n80/model_0080499.pth
#weight_path = weight_dir + "depth_lr4_wo_pretrained/model_final.pth"#depth_n2lr4_output_pretrained2/model_0080499.pth"#"output_RGB_n2/model_final.pth"
#"depth_output_n2_lr5/model_0003999.pth"#"depth_output_n2_lr5/model_0023999n2.pth"
# #"output_RGB_n2/model_final.pth"#"output_RGB/model_final.pth"
#weight_path = "../../Mask2Former/output_RGB_n2/model_final.pth"
#cfg.INPUT.INPUT_IMAGE = 'RGBD_ADD'

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
test_sample(cfg, dataset[3], predictor, visualization=True)
#test_dataset(cfg, ocid_dataset, predictor)
test_dataset(cfg, dataset, predictor, visualization=True)