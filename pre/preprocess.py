import os
import sys

import cv2
import numpy as np
import slideio
from PIL import Image
import subprocess
from collections import Counter
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--save_dir', type=str, help='dir to save csv file')
parser.add_argument('--wsi_path', type=str, help='path to save csv file')
args = parser.parse_args()


def cut_wsi2(wsi_path, b, save_loc=False):
    # save location of sub pic rather than sub pic
    name = wsi_path.split('/')[-1][:-4]
    d = os.path.join(args.save_dir, f'{name}.csv')

    if save_loc and os.path.exists(d):
        return

        # b = 8 * 512
    slide = slideio.open_slide(wsi_path, "SVS")
    scene = slide.get_scene(0)  # scene.size:(W,H)
    # image = scene.read_block(scene.rect, (scene.size[0], scene.size[1]))  # image: H x W x 3
    # im1: H x W x 3
    w, h = scene.size

    m, n = h // b, w // b  # m x n  bag

    sel = [0] * m * n
    cnt_bag = 0

    col = ['bag_id', 'x', 'y']
    data = []
    for i in range(m):  # traverse all bag
        for j in range(n):
            lu = [i * b, j * b]  # left_up (h,w)
            ld = [i * b + b - 1, j * b]
            ru = [i * b, j * b + b - 1]
            rd = [i * b + b - 1, j * b + b - 1]
            bag_loc = [lu, ld, ru, rd]
            bag_id = i * n + j

            bag_image = scene.read_block((lu[1], lu[0], b, b), (b, b))  # image: H x W x 3
            ret, mask = cv2.threshold(bag_image, 127, 255, cv2.THRESH_BINARY)

            # sel[bag_id]= 1 if (mask==255).sum()/len(mask.reshape(-1)) <0.1 else 0
            temp = np.all(mask == 255, axis=2)
            sel[bag_id] = temp.sum() / temp.size

            if sel[bag_id] < 0.98 and save_loc:
                data.append([bag_id, lu[1], lu[0]])

    t = pd.DataFrame(columns=col, data=data)
    t.to_csv(d, index=False)
    return sel, m, n


if __name__ == '__main__':
    cut_wsi2(args.wsi_path, b=2048, save_loc=True)

    # mask, m, n = cut_wsi(sys.argv[1], b=2048, savepng=False)
    # check_cut(mask, m, n, sys.argv[1], b=2048)

    pass
