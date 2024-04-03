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

split_data = False
gen_patches = False

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
    make_patches("train", train_x, train_y, mean, std)
    make_patches("validation", val_x, train_y, mean, std)


train_gen = DataGen(os.path.join(patches_dir, "train"), BATCH_SIZE)
validation_gen = DataGen(os.path.join(patches_dir, "validation"), BATCH_SIZE)



opt = tf.keras.optimizers.Adam(5e-4)
metrics = ["acc","Precision", "Recall", iou, dice_coef, tversky]

# model = mobile_net_model()
# model = resnet50_model()
model = xception_model()
# model = efficient_net_model()

model.compile(loss=focal_tversky_loss, optimizer=opt, metrics=metrics)

mcp_save_loss = ModelCheckpoint("diatom_dataset/min_val_loss.keras",
                                monitor="val_loss", verbose=1, save_best_only=True,
                                save_weights_only=False, mode="min")
mcp_save_dice = ModelCheckpoint("diatom_dataset/max_val_dice.keras",
                                monitor="val_dice_coef", verbose=1, save_best_only=True,
                                save_weights_only=False, mode="max")

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.00001),
    mcp_save_loss,
    mcp_save_dice
]

history = model.fit(train_gen, 
                    epochs = 100, 
                    steps_per_epoch = len(train_gen) // BATCH_SIZE, 
                    batch_size = BATCH_SIZE, 
                    validation_data=validation_gen, 
                    validation_steps = len(validation_gen) // BATCH_SIZE, 
                    callbacks = callbacks)

plt.figure(figsize=(16, 8))
x = np.arange(1,len(history.history['loss'])+1)
plt.plot(x, history.history['loss'], label='Train loss')
plt.plot(x, history.history['val_loss'], label='Validation loss')
plt.plot(x, history.history['dice_coef'], label='Train Dice coef.')
plt.plot(x, history.history['val_dice_coef'], label='Validation Dice coef.')
plt.xlabel('Epoch')
plt.suptitle('Learning curves')
plt.legend()
plt.savefig("model_history.png")

test_x, test_y = get_images(os.path.join(csv_dir, "test.csv"))
evaluate = model.evaluate(test_x, test_y, return_dict = True)
pd.DataFrame([evaluate]).to_csv('evaluate.csv', index=False)

preds = model.predict(test_x[0:10])

fig = plt.figure(figsize=(40,40))
grid = ImageGrid(fig, 111, nrows_ncols=(10, 3), axes_pad=0.1)

for ax, i in zip(grid, range(0, 30)):  
    if i % 3 == 0:
        ax.imshow(test_x[int(i / 3)])
    elif i % 3 == 1:
        ax.imshow(test_y[int(i / 3)], cmap="gray")
    else:
         ax.imshow(preds[int(i / 3)], cmap="gray")

plt.savefig("pred_images.png")