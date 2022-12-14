import numpy as np
import cv2
import keras
import random
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, Input, Concatenate, Activation
import argparse
import os.path
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    dest='batch_size',
                    type=int,
                    default=2,
                    help='Size of images batch per step of learning, default=2')
parser.add_argument('--faces_dir',
                    dest='face_path',
                    type=str,
                    default='CelebAMask-HQ/CelebAMask-HQ-img',
                    help='Input dir for faces images')
parser.add_argument('--masks_dir',
                    dest='mask_path',
                    type=str,
                    default='CelebAMask-HQ/mask',
                    help='Input dir for masks images')
parser.add_argument('--fahand_dir',
                    dest='face_hand_path',
                    type=str,
                    default='CelebAMask-HQ/CelebA-hand',
                    help='Input dir for face-hand images')
parser.add_argument('--mahand_dir',
                    dest='mask_hand_path',
                    type=str,
                    default='CelebAMask-HQ/mask-hand',
                    help='Input dir for mask-hand images')
parser.add_argument('--weights_dir',
                    dest='weights_path',
                    type=str,
                    default='CelebAMask-HQ/weights',
                    help='Output dir for weights')
args = parser.parse_args()

face_path = args.face_path
mask_path = args.mask_path
weights_path = args.weights_path
face_hand_path = args.face_hand_path
mask_hand_path = args.mask_hand_path

path = face_path
files = [f for f in listdir(path) if isfile(join(path, f))]
train_length = round(len(files) * 0.95)
train = files[:train_length]
val = files[train_length:]



def generator_for_keras(files, batch_size):
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size - 1):
            rand = random.choice(files)
            img = cv2.imread(os.path.join(face_path, rand))
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            mask = cv2.imread(os.path.join(mask_path, rand), cv2.IMREAD_GRAYSCALE)

            x_batch += [img]
            y_batch += [mask]

        for i in range(batch_size - 1, batch_size):
            rand = random.randint(0, 4446)
            img = cv2.imread(os.path.join(face_hand_path, str(rand) + '.jpg'))
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            mask = cv2.imread(os.path.join(mask_hand_path, str(rand) + '.jpg'),
                              cv2.IMREAD_GRAYSCALE)

            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch) / 255.

        yield x_batch, y_batch

def model_creation():

    base_model = ResNet50(weights='imagenet', input_shape=[512, 512, 3], include_top=False, classes=1)

    inp = base_model.get_layer('input_1').output

    conv_1 = base_model.get_layer('conv1_relu').output
    conv_2 = base_model.get_layer('conv2_block3_out').output
    conv_3 = base_model.get_layer('conv3_block4_out').output
    conv_4 = base_model.get_layer('conv4_block6_out').output
    conv_5 = base_model.get_layer('conv5_block3_out').output

    up_1 = UpSampling2D(2, interpolation='bilinear')(conv_5)
    conc_1 = Concatenate()([up_1, conv_4])
    conv_conc_1 = Conv2D(1028, (3, 3), padding='same')(conc_1)
    conv_conc_1 = Activation('relu')(conv_conc_1)

    up_2 = UpSampling2D(2, interpolation='bilinear')(conv_conc_1)
    conc_2 = Concatenate()([up_2, conv_3])
    conv_conc_2 = Conv2D(514, (3, 3), padding='same')(conc_2)
    conv_conc_2 = Activation('relu')(conv_conc_2)

    up_3 = UpSampling2D(2, interpolation='bilinear')(conv_conc_2)
    conc_3 = Concatenate()([up_3, conv_2])
    conv_conc_3 = Conv2D(256, (3, 3), padding='same')(conc_3)
    conv_conc_3 = Activation('relu')(conv_conc_3)

    up_4 = UpSampling2D(2, interpolation='bilinear')(conv_conc_3)
    conc_4 = Concatenate()([up_4, conv_1])
    conv_conc_4 = Conv2D(64, (3, 3), padding='same')(conc_4)
    conv_conc_4 = Activation('relu')(conv_conc_4)

    up_5 = UpSampling2D(2, interpolation='bilinear')(conv_conc_4)
    conv_up_5 = Conv2D(1, (3, 3), padding='same')(up_5)
    result = Activation('sigmoid')(conv_up_5)

    return Model(base_model.input, result)

def callbacks_creation(weights_path):

    best_w = keras.callbacks.ModelCheckpoint(os.path.join(weights_path, 'resnet_best.h5'),
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto',
                                            save_freq=500)

    last_w = keras.callbacks.ModelCheckpoint(os.path.join(weights_path, 'resnet_last.h5'),
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=False,
                                            save_weights_only=True,
                                            mode='auto',
                                            save_freq=50)

    return [best_w, last_w]

model = model_creation()

callbacks = callbacks_creation(weights_path)

adam = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(adam, 'binary_crossentropy')

batch_size = args.batch_size
model.fit(generator_for_keras(train, batch_size),
                    steps_per_epoch=1000,
                    epochs=2,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator_for_keras(val, batch_size),
                    validation_steps=50,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0)
