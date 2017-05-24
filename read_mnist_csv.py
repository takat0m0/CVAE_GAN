#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import cv2

def _mod_fig(fig):
    tmp = cv2.resize(fig, (32, 32))
    tmp = np.reshape(tmp, (32, 32, 1))
    ret = tmp/127.5 - 1.0
    return ret

def _make_one_hot(i, class_num):
    ret = [0.0] * class_num
    ret[i] = 1.0
    return ret

def _read_a_line(target_line):
    tmp = np.asarray([float(_) for _ in target_line.strip().split(',')], dtype = np.float32)
    ret_label = int(tmp[0])
    ret_array = np.reshape(tmp[1:], (28, 28, 1))
    return _make_one_hot(ret_label, 10), _mod_fig(ret_array)

def read_file(target_file):
    labels = []
    figs = []
    with open(target_file, 'r') as f:
        for l in f:
            label, array = _read_a_line(l)
            labels.append(label)
            figs.append(array)
    return np.asarray(labels, dtype = np.float32), np.asarray(figs, dtype = np.float32)
            
if __name__ == u'__main__':
    l, a = read_file('mnist_test.csv')
    print(a[0])
