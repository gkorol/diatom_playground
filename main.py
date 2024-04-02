import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, Xception, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import shutil

from dataset import *
from models import *


PATCH_SIZE = 256
NUM_PATCHES = 10

split_data = True
gen_patches = True

if split_data:
    train_images, val_test_images = train_test_split(list(map(lambda f: f.split(".png")[0], os.listdir(images_dir))), train_size = 0.50)
    validation_images, test_images = train_test_split(val_test_images, train_size = 0.30)

    save_images(train_images, "train")
    save_images(validation_images, "validation")
    save_images(test_images, "test")

if gen_patches:
    train_x, train_y = get_images(os.path.join(csv_dir, "train.csv"))
    val_x, val_y = get_images(os.path.join(csv_dir, "validation.csv"))
    mean, std = get_norm_params(train_x)
    print(mean, std)

patches_dir = "/kaggle/input/diatom-patches"
IMAGE_SIZE = 256
BATCH_SIZE = 32


train_gen = DataGen(os.path.join(patches_dir, "train"), BATCH_SIZE)
validation_gen = DataGen(os.path.join(patches_dir, "validation"), BATCH_SIZE)



opt = tf.keras.optimizers.Adam(5e-4)
metrics = ["acc","Precision", "Recall", iou, dice_coef, tversky]

# model = mobile_net_model()
# model = resnet50_model()
model = xception_model()
# model = efficient_net_model()


model.compile(loss=focal_tversky_loss, optimizer=opt, metrics=metrics)
