#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np

from Model import Model
from util import get_figs, dump_figs
from read_mnist_csv import read_file

if __name__ == u'__main__':

    # figs dir
    file_name = u'mnist_test.csv'
    
    # parameter
    batch_size = 100
    pre_epoch_num = 10
    epoch_num = 100
    z_dim = 20
    num_class = 10
    
    # get_data
    print('-- get figs--')
    labels, figs = read_file(file_name)
    assert(len(figs) == len(labels))
    print('num figs = {}'.format(len(figs)))
    
    # make model
    print('-- make model --')
    model = Model(z_dim, batch_size, num_class)
    model.set_model()
    
    
    # training
    print('-- begin training --')
    num_one_epoch = len(figs) //batch_size

    nrr = np.random.RandomState()
    def shuffle(x, y):
        rand_ix = nrr.permutation(x.shape[0])
        return x[rand_ix], y[rand_ix]

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        # -- pre training ---------------
        for epoch in range(pre_epoch_num):
            epoch_figs, epoch_labels = shuffle(figs, labels)

            print('** pre_epoch {} begin **'.format(epoch))
            obj_class, obj_vae, obj_dec, obj_disc = 0.0, 0.0, 0.0, 0.0
            
            for step in range(num_one_epoch):
                
                # get batch data
                batch_figs = epoch_figs[step * batch_size: (step + 1) * batch_size]
                batch_labels = epoch_labels[step * batch_size: (step + 1) * batch_size]
                batch_z = np.random.randn(batch_size, z_dim)
                
                # train
                obj_class += model.pretraining_class(sess, batch_figs, batch_labels)
                obj_disc += model.pretraining_disc(sess, batch_figs, batch_labels,  batch_z)
                obj_vae += model.pretraining_vae(sess, batch_figs, batch_labels)
                obj_dec += model.pretraining_dec(sess, batch_figs, batch_labels, batch_z)
                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    tmp_z = model.encoding(sess, batch_figs, batch_labels)
                    tmp_figs = model.gen_fig(sess, batch_labels, tmp_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result')
                    
            print('epoch:{}, v_obj = {}, dec_obj = {}, disc_obj = {}'.format(epoch,
                                                                        obj_vae/num_one_epoch,
                                                            obj_dec/num_one_epoch,
                                                            obj_disc/num_one_epoch))
            saver.save(sess, './model.dump')
            
        # -- main training ---------------            
        for epoch in range(epoch_num):
            epoch_figs, epoch_labels = shuffle(figs, labels)
            
            print('** epoch {} begin **'.format(epoch))
            obj_class, obj_vae, obj_dec, obj_disc = 0.0, 0.0, 0.0, 0.0
            
            for step in range(num_one_epoch):
                
                # get batch data
                batch_figs = epoch_figs[step * batch_size: (step + 1) * batch_size]
                batch_labels = epoch_labels[step * batch_size: (step + 1) * batch_size]                
                batch_z = np.random.randn(batch_size, z_dim)
                # train
                obj_class += model.training_class(sess, batch_figs, batch_labels)
                obj_disc += model.training_disc(sess, batch_figs, batch_labels, batch_z)
                obj_vae += model.training_vae(sess, batch_figs, batch_labels)
                obj_dec += model.training_dec(sess, batch_figs, batch_labels, batch_z)

                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    tmp_z = model.encoding(sess, batch_figs, batch_labels)
                    tmp_figs = model.gen_fig(sess, batch_labels, tmp_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result2')

                    
            print('epoch:{}, v_obj = {}, dec_obj = {}, disc_obj = {}'.format(epoch,
                                                                        obj_vae/num_one_epoch,
                                                            obj_dec/num_one_epoch,
                                                            obj_disc/num_one_epoch))
            saver.save(sess, './model.dump')
