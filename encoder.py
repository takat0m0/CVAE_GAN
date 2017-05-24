#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases, get_dim, lrelu, conv_layer, linear_layer
from batch_normalize import batch_norm

class Encoder(object):
    def __init__(self, layer_chanels, fc_dim, z_dim):
        self.layer_chanels = layer_chanels
        self.fc_dim = fc_dim
        self.z_dim = z_dim
        
        self.name_scope_conv = u'enc_conv'
        self.name_scope_label = u'enc_label'        
        self.name_scope_fc = u'enc_fc'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_conv in var.name or self.name_scope_fc in var.name or self.name_scope_label in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, figs, labels, is_training, reuse = False):
        fig_shape = figs.get_shape().as_list()
        height, width = fig_shape[1:3]
        class_num = get_dim(labels)
        with tf.variable_scope(self.name_scope_label, reuse = reuse):
            tmp = linear_layer(labels, class_num, height * width, 'reshape')
            tmp = tf.reshape(tmp, [-1, height, width, 1])
        h = tf.concat((figs, tmp), 3)
        
        # convolution
        with tf.variable_scope(self.name_scope_conv, reuse = reuse):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):

                conved = conv_layer(inputs = h,
                                    out_num = out_chan,
                                    filter_width = 5, filter_hight = 5,
                                    stride = 2, l_id = i)
                
                if i == 0:
                    h = tf.nn.relu(conved)
                    #h = lrelu(conved)
                else:
                    bn_conved = batch_norm(conved, i, is_training)
                    h = tf.nn.relu(bn_conved)
                    #h = lrelu(bn_conved)
        # full connect
        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])
        
        with tf.variable_scope(self.name_scope_fc, reuse = reuse):
            h = linear_layer(h, dim, self.fc_dim, 'fc')
            h = batch_norm(h, 'en_fc_bn', is_training)
            h = tf.nn.relu(h)

            mu = linear_layer(h, self.fc_dim, self.z_dim, 'mu')
            log_sigma = linear_layer(h, self.fc_dim, self.z_dim, 'sigma')
            
        return mu, log_sigma
    
if __name__ == u'__main__':
    g = Encoder([3, 64, 128, 256], 2048, 512)
    figs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    labels = tf.placeholder(tf.float32, [None, 200])
    g.set_model(figs, labels, True)
