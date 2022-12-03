import numpy as np
import cv2
import keras
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, Input, Concatenate, Activation
import argparse
from os import listdir
from os.path import isfile, join
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path',
                    dest='w_path',
                    type=str,
                    default='CelebAMask-HQ/weights/resnet_last.h5',
                    help='Input path to the weights')
parser.add_argument('--images_path',
                    dest='im_path',
                    type=str,
                    help='Input path to the image or images')
parser.add_argument('--save_path',
                    dest='save_path',
                    type=str,
                    help='Output path for segmentation')
args = parser.parse_args()

w_path = args.w_path
im_path = args.im_path
save_path = args.save_path

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

model = Model(base_model.input, result)
model.load_weights(w_path)

files = [f for f in listdir(im_path) if isfile(join(im_path, f))]
shape_before = []
prep_images = []
for i in files:
    image = cv2.imread(os.path.join(im_path, i))
    shape_before.append([image.shape[0], image.shape[1]])
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = image/255.
    prep_images.append(image)
data = np.array(prep_images)

predict = model.predict(data)
for i in range(len(predict)):
    helper = predict[i, ..., 0] > 0.5
    helper = helper * 255.0
    helper = cv2.resize(helper, (shape_before[i][1], shape_before[i][0]), interpolation=cv2.INTER_LINEAR)
    helper[helper < 255] = 0
    cv2.imwrite(os.path.join(save_path, files[i]), helper)


