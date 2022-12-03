import os.path as osp
import os
import cv2
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--masks_parts',
                    dest='face_mask',
                    type=str,
                    default='CelebAMask-HQ/CelebAMask-HQ-mask-anno',
                    help='Input dir for masks parts')
parser.add_argument('--output_mask',
                    dest='mask_path',
                    type=str,
                    default='CelebAMask-HQ/mask',
                    help='Output dir for masked images')
args = parser.parse_args()

face_sep_mask = args.face_mask
mask_path = args.mask_path

counter = 0
total = 0
for i in range(15):

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    toone = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'mouth', 'u_lip', 'l_lip']
    tozero = ['eye_g', 'ear_r', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                if att in toone:
                    mask[sep_mask == 225] = 255
                else:
                    mask[sep_mask == 225] = 0
        cv2.imwrite('{}/{}.jpg'.format(mask_path, j), mask)
print(counter, total)