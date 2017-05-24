#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import cv2
import numpy as np

def get_weights(name, shape, stddev, trainable = True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer = tf.random_normal_initializer(stddev = stddev),
                           trainable = trainable)

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)

def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim *= d
    return dim

def get_figs(dir_name):
    ret = []
    for file_name in os.listdir(dir_name):
        tmp = cv2.imread(os.path.join(dir_name, file_name))
        if tmp is None:
            continue
        ret.append(tmp/127.5 - 1.0)
        break
    return np.asarray(ret, dtype = np.float32)

def dump_figs(imgs, dir_name):
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dir_name, '{}.jpg'.format(i)), (img + 1.0) * 127.5)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def conv_layer(inputs, out_num, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, in_chanel, out_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, inputs.get_shape()[-1], out_num],
                          0.02)
    
    biases = get_biases(l_id, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)


def deconv_layer(inputs, out_shape, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, out_shape[-1], inputs.get_shape()[-1]],
                          0.02)
    
    biases = get_biases(l_id, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    return tf.nn.bias_add(deconved, biases)

def linear_layer(inputs, in_dim, out_dim, name):
    w = get_weights(name, [in_dim, out_dim], 0.02)
    b = get_biases(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b
