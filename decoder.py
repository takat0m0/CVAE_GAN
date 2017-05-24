#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import lrelu, deconv_layer, linear_layer
from batch_normalize import batch_norm


class Decoder(object):
    def __init__(self, z_dim, layer_chanels):
        self.z_dim = z_dim
        self.in_dim = 4
        self.layer_chanels = layer_chanels

        self.name_scope_reshape = u'dec_reshape_z'
        self.name_scope_label = u'dec_label'
        self.name_scope_deconv = u'dec_deconvolution'

    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_deconv in var.name or self.name_scope_reshape in var.name or self.name_scope_label in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, z, labels, batch_size, is_training, reuse = False):

        # reshape z
        with tf.variable_scope(self.name_scope_reshape, reuse = reuse):
            h = linear_layer(z, self.z_dim, self.in_dim * self.in_dim * self.layer_chanels[0], 'reshape')
            h = batch_norm(h, 'reshape', is_training)
            h = lrelu(h)
            
        h_z = tf.reshape(h, [-1, self.in_dim, self.in_dim, self.layer_chanels[0]])            
        # reshape labels
        with tf.variable_scope(self.name_scope_label, reuse = reuse):
            h = linear_layer(z, self.z_dim, self.in_dim * self.in_dim * self.layer_chanels[0], 'label')
            h = batch_norm(h, 'label', is_training)
            h = lrelu(h)
            
        # concat
        h_label = tf.reshape(h, [-1, self.in_dim, self.in_dim, self.layer_chanels[0]])
        h = tf.concat([h_z, h_label], 3)
        
        # deconvolution
        layer_num = len(self.layer_chanels) - 1
        with tf.variable_scope(self.name_scope_deconv, reuse = reuse):
            for i, (in_chan, out_chan) in enumerate(zip(self.layer_chanels, self.layer_chanels[1:])):
                deconved = deconv_layer(inputs = h,
                                        out_shape = [batch_size, self.in_dim * 2 ** (i + 1), self.in_dim * 2 **(i + 1), out_chan],
                                        filter_width = 5, filter_hight = 5,
                                        stride = 2, l_id = i)
                if i == layer_num -1:
                    h = tf.nn.tanh(deconved)
                else:
                    bn_deconved = batch_norm(deconved, i, is_training)
                    #h = tf.nn.relu(bn_deconved)
                    h = lrelu(bn_deconved)

        return h
        
    
if __name__ == u'__main__':
    g = Decoder(512, [256, 128, 32, 3])
    z = tf.placeholder(tf.float32, [None, 512])
    labels = tf.placeholder(tf.float32, [None, 200])    
    g.set_model(z, labels, 100, True)
