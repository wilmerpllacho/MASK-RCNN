
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import albumentations as A
import sys
import load_data as data

from mrcnn import visualize2
from mrcnn.config import Config
from mrcnn import model as modellib, utils
#from mrcnn import Mask_RCNN as Mask_RCNN
import Dataset as Dataset

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR = './'

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_lug_0100.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

transform = A.Compose([
                A.Rotate(p = 0.2, limit = 30, border_mode=0), # Please select an interpolation method
                A.HorizontalFlip(p = 0.2),
                A.OneOf([
                    A.GridDistortion(p = 0.1, distort_limit = 0.2, border_mode=0),
                    A.ElasticTransform(sigma = 10, alpha = 1,  p = 0.1, border_mode=0)
                ])
            ])

path_images='dataset/IMAGE/'
path_masks='dataset/MASK/'

dataset_image = []
dataset_mask  = []

images_traing,masks_traing,images_validate,masks_validate=data.split_dataset(path_images,path_masks,0.9)
images_generate, masks_generate = data.generate_dataset(images_traing,masks_traing,transform,0.5)

#images_traing.extend(images_generate)
#masks_traing.extend(masks_generate)

training_dataset   =    data.generate_dataset_mask(images_traing,masks_traing)
val_dataset        =    data.generate_dataset_mask(images_validate,masks_validate)

dataset_train = Dataset.Dataset()
dataset_train.load_lug(training_dataset)
dataset_train.prepare()

# Validation dataset
dataset_val = Dataset.Dataset()
dataset_val.load_lug(val_dataset)
dataset_val.prepare()

print(len(training_dataset))
class LugConfig(Config):
    NAME = 'lug segmentation'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # background + 1 (lug)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 5
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    BACKBONE = 'resnet50'

    POST_NMS_ROIS_INFERENCE = 1000 
    POST_NMS_ROIS_TRAINING = 2000

    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
config = LugConfig()


# Create model in training mode

model_train = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
#COCO_MODEL_PATH = model_train.find_last()
model_train.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model_train.train(dataset_train, dataset_val, learning_rate=0.00001, epochs=41, layers='heads')


model_train = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
#COCO_MODEL_PATH = model_train.find_last()
model_train.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model_train.train(dataset_train, dataset_val, learning_rate=0.00001, epochs=41, layers='heads')

model_train = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
#COCO_MODEL_PATH = model_train.find_last()
model_train.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model_train.train(dataset_train, dataset_val, learning_rate=0.00001, epochs=41, layers='heads')

model_train = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
#COCO_MODEL_PATH = model_train.find_last()
model_train.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model_train.train(dataset_train, dataset_val, learning_rate=0.00001, epochs=41, layers='heads')

model_train = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
#COCO_MODEL_PATH = model_train.find_last()
model_train.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model_train.train(dataset_train, dataset_val, learning_rate=0.00001, epochs=41, layers='heads')
