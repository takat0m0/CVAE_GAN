#! -*- coding:utf-8 -*-

import cv2
import os
import sys
import numpy as np

def _change_one_hot(i, maxnum):
    tmp = [0.0] * maxnum
    tmp[i - 1] = 1.0
    return np.asarray(tmp, dtype = np.float32)

def get_figs_labels(fig_dir, label_file):
    label_tmp = []
    with open(label_file, 'r') as f:
        for l in f:
            label_tmp.append(int(l.strip()))
    maxnum = max(label_tmp)
    label_ret = [_change_one_hot(_, maxnum) for _ in label_tmp]

    imgs = []
    for filename in os.listdir(fig_dir):
        image_file = os.path.join(fig_dir, filename)
        img = cv2.imread(image_file)
        img_ = cv2.resize(img, (128, 128))
        img = img_/127.5 - 1.0
        imgs.append(img)
    return np.asarray(imgs, dtype = np.float32), np.asarray(label_ret, dtype = np.float32)

if __name__ == u'__main__':
    imgs, labels = get_figs_labels('jpg', 'label.txt')
    print(imgs[0])
