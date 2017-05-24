#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases, get_dim, lrelu, conv_layer
from batch_normalize import batch_norm

class Discriminator(object):
    def __init__(self, layer_chanels, fc_dim):
        self.layer_chanels = layer_chanels
        self.fc_dim = fc_dim
        
        self.name_scope_conv = u'dis_conv'
        self.name_scope_fc = u'dis_fc'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_conv in var.name or self.name_scope_fc in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, figs, is_training, reuse = False):
        # return only logits
        
        h = figs
        
        # convolution
        with tf.variable_scope(self.name_scope_conv, reuse = reuse):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):
                if i == 0:
                    conved = conv_layer(inputs = h,
                                        out_num = out_chan,
                                        filter_width = 5, filter_hight = 5,
                                        stride = 1, l_id = i)

                    h = tf.nn.relu(conved)
                    #h = lrelu(conved)
                else:
                    conved = conv_layer(inputs = h,
                                        out_num = out_chan,
                                        filter_width = 5, filter_hight = 5,
                                        stride = 2, l_id = i)

                    bn_conved = batch_norm(conved, i, is_training)
                    h = tf.nn.relu(bn_conved)
                    #h = lrelu(bn_conved)
                    
        feature_image = h
        
        # full connect
        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])
        
        with tf.variable_scope(self.name_scope_fc, reuse = reuse):
            weights = get_weights('fc', [dim, self.fc_dim], 0.02)
            biases  = get_biases('fc', [self.fc_dim], 0.0)
            h = tf.matmul(h, weights) + biases
            h = batch_norm(h, 'fc', is_training)
            h = tf.nn.relu(h)

            weights = get_weights('fc2', [self.fc_dim, 1], 0.02)
            biases  = get_biases('fc2', [1], 0.0)
            h = tf.matmul(h, weights) + biases
            
        return h, feature_image
        #return tf.nn.sigmoid(h)
    
if __name__ == u'__main__':
    g = Discriminator([3, 32, 128, 256, 256], 512)
    figs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    g.set_model(figs, True)
