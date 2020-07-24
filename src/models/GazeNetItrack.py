#!/usr/bin/env python3

# using http://cs231n.stanford.edu/reports/2017/pdfs/229.pdf
# and https://arxiv.org/pdf/1711.09017.pdf Gazenet

import tensorflow as tf
import util.gaze
import parameters
import math as m
import sys


basefiltf = 64
basefilte = 32
base_filt = 16
class ITRACK():
#eyes 224x60 # 3 2 2 -> 22x5
#face 224x224 # 2 2 2 #doing 3 first as well -> 22x22
#dilation_rate = (1,1)


#tf.layers.separable_conv2d


#TODO try initializer for separable conv2d
    def xception_f(self, tensor, name, reuse=None, is_training=True):

        #BEGIN ENTRY FLOW
        print('in output shape', tensor.get_shape())
        conv = tf.layers.conv2d(tensor, basefiltf, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv1',reuse=reuse)#self.conv3by3(tensor,name+'_C1',2, 32)

        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        print('conv output shape', conv.get_shape())

        conv2 = tf.layers.conv2d(conv, basefiltf*2, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        print('conv2 output shape', conv2.get_shape())
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        print('conv3 output shape', conv3.get_shape())
        conv4 = tf.layers.separable_conv2d(conv3, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv4',reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        print('conv4 output shape', conv4.get_shape())
        max1 = tf.layers.max_pooling2d(conv4, pool_size=3,strides=2, padding='same',
                              data_format="channels_first")
        print('max1 output shape', max1.get_shape())
        convs1 = tf.layers.conv2d(conv2, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv1',reuse=reuse)
        convs1 = tf.layers.batch_normalization(convs1, training=is_training, axis=1) #applying BN before relu here and afterwards since it's still a topic of debate, I will assume minimal impact
        print('convs1 output shape', convs1.get_shape())
        l1 = tf.concat([max1,convs1],1)

        l1_b = tf.nn.relu(l1)

        conv5 = tf.layers.separable_conv2d(l1_b, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv6',reuse=reuse)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training, axis=1)
        max2 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs2 = tf.layers.conv2d(l1, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv2',reuse=reuse)
        convs2 = tf.layers.batch_normalization(convs2, training=is_training, axis=1)
        l2 = tf.concat([max2, convs2],1)

        l2_b = tf.nn.relu(l2)

        conv7 = tf.layers.separable_conv2d(l2_b, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv7',reuse=reuse)
        conv7 = tf.layers.batch_normalization(conv7, training=is_training, axis=1)
        conv8 = tf.layers.separable_conv2d(conv7, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv8',reuse=reuse)
        conv8 = tf.layers.batch_normalization(conv8, training=is_training, axis=1)
        max3 = tf.layers.max_pooling2d(conv8, pool_size=4, strides=2,padding='same',
                              data_format="channels_first")

        convs3 = tf.layers.conv2d(l2, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv3',reuse=reuse)
        convs3 = tf.layers.batch_normalization(convs3, training=is_training, axis=1)

        l3 = tf.concat([max3, convs3],1)

        #END ENTRY FLOW

        #BEGIN MIDDLE FLOW

        M1 = self.Middle_flow(l3, name+'_M1', reuse=reuse, is_training=is_training)
        M2 = self.Middle_flow(M1, name + '_M2', reuse=reuse, is_training=is_training)
        M3 = self.Middle_flow(M2, name + '_M3', reuse=reuse, is_training=is_training)
        M4 = self.Middle_flow(M3, name + '_M4', reuse=reuse, is_training=is_training)
        M5 = self.Middle_flow(M4, name + '_M5', reuse=reuse, is_training=is_training)
        M6 = self.Middle_flow(M5, name + '_M6', reuse=reuse, is_training=is_training)
        M7 = self.Middle_flow(M6, name + '_M7', reuse=reuse, is_training=is_training)
        M8 = self.Middle_flow(M7, name + '_M8', reuse=reuse, is_training=is_training)

        #END MIDDLE FLOW

        #BEGIN END FLOW

        l4 = tf.nn.relu(M8)

        convend1 = tf.layers.separable_conv2d(l4, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_convend5',reuse=reuse)
        convend1 = tf.layers.batch_normalization(convend1, training=is_training, axis=1)
        convend2 = tf.layers.separable_conv2d(convend1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_convend6',reuse=reuse)
        convend2 = tf.layers.batch_normalization(convend2, training=is_training, axis=1)
        max3 = tf.layers.max_pooling2d(convend2, pool_size=3, strides=2, padding='same',
                              data_format="channels_first")

        convsend3 = tf.layers.conv2d(M8, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consvend3',reuse=reuse)
        convsend3 = tf.layers.batch_normalization(convsend3, training=is_training, axis=1)
        l5 = tf.concat([max3, convsend3],1)

        conve1 = tf.layers.separable_conv2d(l5, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end

    def xception_e(self, tensor, name, reuse=None, is_training=True):

        #BEGIN ENTRY FLOW

        conv = tf.layers.conv2d(tensor, basefiltf, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv1',reuse=reuse)#self.conv3by3(tensor,name+'_C1',2, 32)

        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)


        conv2 = tf.layers.conv2d(conv, basefiltf*2, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv4 = tf.layers.separable_conv2d(conv3, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv4',reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)

        max1 = tf.layers.max_pooling2d(conv4, pool_size=3, strides=3,padding='same',
                              data_format="channels_first")

        convs1 = tf.layers.conv2d(conv2, 1, kernel_size=1, strides=3, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv1',reuse=reuse)
        convs1 = tf.layers.batch_normalization(convs1, training=is_training, axis=1) #applying BN before relu here and afterwards since it's still a topic of debate, I will assume minimal impact
        l1 = tf.concat([max1,convs1],1)

        l1_b = tf.nn.relu(l1)

        conv5 = tf.layers.separable_conv2d(l1_b, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv6',reuse=reuse)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training, axis=1)
        max2 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs2 = tf.layers.conv2d(l1, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv2',reuse=reuse)
        convs2 = tf.layers.batch_normalization(convs2, training=is_training, axis=1)
        l2 = tf.concat([max2, convs2],1)

        # Can't work for the eyes
        # l2_b = tf.nn.relu(l2)
        #
        # conv7 = tf.layers.separable_conv2d(l2_b, 728, kernel_size=[3, 3], strides=2, padding='same',
        #                         activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                         data_format="channels_first",name=name+'_conv7',reuse=reuse)
        # conv7 = tf.layers.batch_normalization(conv7, training=is_training, axis=1)
        # conv8 = tf.layers.separable_conv2d(conv7, 728, kernel_size=[3, 3], strides=3, padding='same',
        #                         kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                         data_format="channels_first",name=name+'_conv8',reuse=reuse)
        # conv8 = tf.layers.batch_normalization(conv8, training=is_training, axis=1)
        # max3 = tf.layers.max_pooling2d(conv8, kernel_size=4, strides=2,
        #                       data_format="channels_first")
        #
        # convs3 = tf.layers.conv2d(conv7, 1, kernel_size=1, strides=2, padding='valid',
        #                         kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                         data_format="channels_first",name=name+'_consv3',reuse=reuse)
        # convs3 = tf.layers.batch_normalization(convs3, training=is_training, axis=1)
        #
        # l3 = tf.concat([max3, convs3])

        #END ENTRY FLOW

        #BEGIN MIDDLE FLOW

        M1 = self.Middle_flow(l2 ,name+'_M1', reuse=reuse, is_training=is_training)
        M2 = self.Middle_flow(M1, name + '_M2', reuse=reuse, is_training=is_training)
        M3 = self.Middle_flow(M2, name + '_M3', reuse=reuse, is_training=is_training)
        M4 = self.Middle_flow(M3, name + '_M4', reuse=reuse, is_training=is_training)
        M5 = self.Middle_flow(M4, name + '_M5', reuse=reuse, is_training=is_training)
        M6 = self.Middle_flow(M5, name + '_M6', reuse=reuse, is_training=is_training)
        M7 = self.Middle_flow(M6, name + '_M7', reuse=reuse, is_training=is_training)
        M8 = self.Middle_flow(M7, name + '_M8', reuse=reuse, is_training=is_training)

        #END MIDDLE FLOW

        #BEGIN END FLOW

        conve1 = tf.layers.separable_conv2d(M8, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end

    def xception_er(self, tensor, name, reuse=None, is_training=True):

        #BEGIN ENTRY FLOW

        conv = tf.layers.conv2d(tensor, basefilte, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv1',reuse=reuse)#self.conv3by3(tensor,name+'_C1',2, 32)

        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)


        conv2 = tf.layers.conv2d(conv, basefilte*2, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefilte*4, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv4 = tf.layers.separable_conv2d(conv3, basefilte*4, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv4',reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)

        max1 = tf.layers.max_pooling2d(conv4, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs1 = tf.layers.conv2d(conv2, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv1',reuse=reuse)
        convs1 = tf.layers.batch_normalization(convs1, training=is_training, axis=1) #applying BN before relu here and afterwards since it's still a topic of debate, I will assume minimal impact
        l1 = tf.concat([max1,convs1],1)

        l1_b = tf.nn.relu(l1)

        conv5 = tf.layers.separable_conv2d(l1_b, basefilte*6, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefilte*6, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv6',reuse=reuse)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training, axis=1)
        max2 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs2 = tf.layers.conv2d(l1, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv2',reuse=reuse)
        convs2 = tf.layers.batch_normalization(convs2, training=is_training, axis=1)
        l2 = tf.concat([max2, convs2],1)


        #END ENTRY FLOW

        #BEGIN MIDDLE FLOW

        M1 = self.Middle_flow(l2,name+'_M1', reuse=reuse, is_training=is_training)
        M2 = self.Middle_flow(M1, name + '_M2', reuse=reuse, is_training=is_training)
        M3 = self.Middle_flow(M2, name + '_M3', reuse=reuse, is_training=is_training)
        M4 = self.Middle_flow(M3, name + '_M4', reuse=reuse, is_training=is_training)
        M5 = self.Middle_flow(M4, name + '_M5', reuse=reuse, is_training=is_training)
        M6 = self.Middle_flow(M5, name + '_M6', reuse=reuse, is_training=is_training)
        M7 = self.Middle_flow(M6, name + '_M7', reuse=reuse, is_training=is_training)
        M8 = self.Middle_flow(M7, name + '_M8', reuse=reuse, is_training=is_training)

        #END MIDDLE FLOW

        #BEGIN END FLOW

        conve1 = tf.layers.separable_conv2d(M8, basefilte*8, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefilte*12, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end


    def Middle_flow(self, tensor, name, reuse=None, is_training=True):

        m1 = tf.nn.relu(tensor)
        conv1 = tf.layers.separable_conv2d(m1, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conv1', reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)
        conv2 = tf.layers.separable_conv2d(conv1, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conv2', reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                           data_format="channels_first", name=name + '_conv3', reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3 = tf.concat([conv3,tensor],1)
        return conv3

    def conv3by3(self,tensor,name,filters,stride=1,reuse=None): #can be sobel filters

        conv3 = tf.layers.conv2d(tensor, filters, kernel_size=[1, 3], strides=stride, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_1',reuse=reuse)

        conv3 = tf.layers.conv2d(conv3, filters, kernel_size=[3, 1], strides=stride, padding='same',
                                 activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                 data_format="channels_first",name=name+'_2',reuse=reuse)

        return conv3

    def inception_module(self,tensor,name, filters,reuse=None, is_training=True):

        conv = tf.layers.conv2d(tensor, filters, kernel_size=[1,1], strides=[1,1], padding='same',
                            kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                            data_format='channels_first',name=name+'_inc1',reuse=reuse)

        conv3 = self.conv3by3(conv,name+'_conv3_', filters,reuse=reuse)

        conv5 = self.conv3by3(conv,name+'_conv5_1_', filters,reuse=reuse)
        conv5 = self.conv3by3(conv5,name+'_conv5_2_', filters,reuse=reuse)

        conv7 = self.conv3by3(conv,name+'_conv7_1_', filters,reuse=reuse)
        conv7 = self.conv3by3(conv7,name+'_conv7_2_', filters,reuse=reuse)
        conv7 = self.conv3by3(conv7,name+'_conv7_3_', filters, reuse=reuse)

        conv = tf.concat([conv, conv3, conv5, conv7],1)
        #conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
        #conv = tf.nn.relu(conv)
        return conv

    def downsample(self, tensor,name, filters, kernel=[2,2], is_training=True):#level descend is valid conv 3x3x1 and max pool

        conv = tf.layers.conv2d(tensor, filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")


        conv2 = tf.layers.conv2d(conv, filters, kernel_size=kernel, strides=kernel, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")


        max = tf.layers.max_pooling2d(tensor, kernel, kernel,
                                      data_format="channels_first")
        cmax = tf.layers.conv2d(max, filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")

        conv = tf.concat([conv2,cmax],1)
        #conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        return conv
    def gaussian_kernel(self,size, mean, std):

        d = tf.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
        gauss_kernel = tf.einsum('i,j->ij',vals,vals)
        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def apply_filter(self,tensor,gkernel,name='Filt',reuse=None):
        #https://arxiv.org/pdf/1903.07296.pdf
        #check with gaussian blur first aswell
        #blurB = tf.nn.conv2d(tensor[:,tf.newaxis,0,:,:], gkernel, strides=[1, 1, 1, 1], padding="SAME",data_format="NCHW")
        #blurG = tf.nn.conv2d(tensor[:,tf.newaxis, 1, :, :], gkernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
        #blurR = tf.nn.conv2d(tensor[:,tf.newaxis, 2, :, :], gkernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW")
        #blur = tf.concat([blurB,blurG,blurR],1)
        sob = tf.image.sobel_edges(tensor)
        sob = tf.math.sqrt(tf.reduce_sum(tf.math.pow(sob,2),4))
        sobm = tf.reduce_max(sob,[2, 3],keepdims=True)
        sob = sob/(sobm)
        #conv = self.inception_module(tensor,name+'_pre1',filters=32,reuse=reuse,is_training=self.training)#for preprocessing
        #conv = self.conv3by3(tensor, name + '_conv1_', 64, reuse=reuse)#edge style filters
        #conv = self.conv3by3(conv, name + '_conv2_', 64, reuse=reuse)#edge style filters
        ten = tf.concat([sob,tensor], 1)#conv

        return ten

    def conv_batch_relu(self, tensor, filters, kernel=[3, 3], stride=[1, 1], is_training=True, name='convbr',reuse=None):
        conv = tf.layers.conv2d(tensor, filters, kernel_size=kernel, strides=stride, padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first", name=name,reuse=reuse)
        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        conv = tf.nn.relu(conv)
        return conv


    def Unet(self, tensor, name='Unet', reuse=None):
        conv_0_1 = self.conv_batch_relu(tensor, base_filt, is_training=self.training, name=name+'_conv01',reuse=reuse)  # self.model_input
        conv_0_2 = self.conv_batch_relu(conv_0_1, base_filt * 2, is_training=self.training, name=name+'_conv02',reuse=reuse)
        # Level one
        max_1_1 = tf.layers.max_pooling2d(conv_0_2, [5, 5], [5, 5],
                                          data_format="channels_first")  # Stride, Kernel previously [2,2,2]
        # pool_size:An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width)

        conv_1_1 = self.conv_batch_relu(max_1_1, base_filt * 2, is_training=self.training, name=name+'_conv11',reuse=reuse)
        conv_1_2 = self.conv_batch_relu(conv_1_1, base_filt * 4, is_training=self.training, name=name+'_conv12',reuse=reuse)
        # conv_1_2 = tf.layers.dropout(conv_1_2, rate=self.drop, training=self.training)
        # Level two
        max_2_1 = tf.layers.max_pooling2d(conv_1_2, [3, 3], [3, 3], #2,2
                                          data_format="channels_first")  # Stride, Kernel previously [2,2,2]
        conv_2_1 = self.conv_batch_relu(max_2_1, base_filt * 4, is_training=self.training, name=name+'_conv21',reuse=reuse)
        conv_2_2 = self.conv_batch_relu(conv_2_1, base_filt * 8, is_training=self.training, name=name+'_conv22',reuse=reuse)
        # conv_2_2 = tf.layers.dropout(conv_2_2, rate=self.drop, training=self.training)
        # Level three
        max_3_1 = tf.layers.max_pooling2d(conv_2_2, [2, 2], [2, 2],
                                          data_format="channels_first")  # Stride, Kernel previously [2,2,2]
        conv_3_1 = self.conv_batch_relu(max_3_1, base_filt * 8, is_training=self.training, name=name+'_conv31',reuse=reuse)
        conv_3_2 = self.conv_batch_relu(conv_3_1, base_filt * 16, is_training=self.training, name=name+'_conv32',reuse=reuse)
        # conv_3_2 = tf.layers.dropout(conv_3_2, rate=self.drop, training=self.training)
        # Level two
        up_conv_3_2 = tf.layers.conv2d_transpose(conv_3_2, base_filt*16, kernel_size=[2,2], strides=[2,2], padding='same',
                                          use_bias=False,
                                          kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                          data_format="channels_first", name=name+'_uconv32',reuse=reuse)
        #up_conv_3_2 = self.upconvolve(conv_3_2, base_filt * 16)  # , kernel=2,
        # stride=[1, 2, 2])  # Stride previously [2,2,2]
        #concat_2_1 = self.centre_crop_and_concat(conv_2_2, up_conv_3_2)
        concat_2_1 = tf.concat([conv_2_2,up_conv_3_2],1)
        conv_2_3 = self.conv_batch_relu(concat_2_1, base_filt * 8, is_training=self.training, name=name+'_conv23',reuse=reuse)
        conv_2_4 = self.conv_batch_relu(conv_2_3, base_filt * 8, is_training=self.training, name=name+'_conv24',reuse=reuse)
        # conv_2_4 = tf.layers.dropout(conv_2_4, rate=self.drop, training=self.training)
        # Level one
        up_conv_2_1 = tf.layers.conv2d_transpose(conv_2_4, base_filt*8, kernel_size=[3,3], strides=[3,3], padding='same',
                                          use_bias=False,
                                          kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                          data_format="channels_first", name=name+'_uconv21',reuse=reuse)
        #up_conv_2_1 = self.upconvolve(conv_2_4, base_filt * 8)  # , kernel=2,
        # stride=[1, 2, 2])  # Stride previously [2,2,2]
        #concat_1_1 = self.centre_crop_and_concat(conv_1_2, up_conv_2_1)
        concat_1_1 = tf.concat([conv_1_2, up_conv_2_1], 1)
        conv_1_3 = self.conv_batch_relu(concat_1_1, base_filt * 4, is_training=self.training, name=name+'_conv13',reuse=reuse)
        conv_1_4 = self.conv_batch_relu(conv_1_3, base_filt * 4, is_training=self.training, name=name+'_conv14',reuse=reuse)
        # conv_1_4 = tf.layers.dropout(conv_1_4, rate=self.drop, training=self.training)
        # Level zero
        up_conv_1_0 = tf.layers.conv2d_transpose(conv_1_4, base_filt*4, kernel_size=[5,5], strides=[5,5], padding='same',
                                          use_bias=False,
                                          kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                          data_format="channels_first", name=name+'_uconv10',reuse=reuse)
        #up_conv_1_0 = self.upconvolve(conv_1_4, base_filt * 4)  # , kernel=2,
        # stride=[1, 2, 2])  # Stride previously [2,2,2]
        #concat_0_1 = self.centre_crop_and_concat(conv_0_2, up_conv_1_0)
        concat_0_1 = tf.concat([conv_0_2, up_conv_1_0], 1)
        conv_0_3 = self.conv_batch_relu(concat_0_1, base_filt * 2, is_training=self.training, name=name+'_conv03',reuse=reuse)
        conv_0_4 = self.conv_batch_relu(conv_0_3, base_filt * 2, is_training=self.training, name=name+'_conv04',reuse=reuse)


        conv_0_5 = tf.layers.conv2d(conv_0_4, 1, [1, 1], [1, 1], padding='same',
                                    data_format="channels_first", name=name+'_conv05',reuse=reuse)  # 1 instead of OUTPUT_CLASSES
        return conv_0_5


    def __init__(self, drop=0.3):#, learning_rate=parameters.LEARNING_RATE):
        self.f = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
        self.er = tf.placeholder(tf.float32, shape=(None, 3, 60, 224))
        self.le = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.re = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.h = tf.placeholder(tf.float32, shape=(None, 2))
        self.fl = tf.placeholder(tf.float32, shape=(None, 33, 2))
        self.g = tf.placeholder(tf.float32, shape=(None, 2))
        self.drop = drop
        self.training = tf.placeholder(tf.bool)
        self.LR = tf.placeholder(tf.float32)
        learning_rate=self.LR



        gauss_kernel = self.gaussian_kernel(2,0.0,1.0)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        #gauss_kernel = tf.concat([gauss_kernel,gauss_kernel,gauss_kernel],1)


        pi = tf.constant(m.pi)

        # However both of these are done by batch normalization to a certain degree + noisy dataset
        # Xavier init. uses mean 0 and variance of 1/Neurons (or Navg, being the average of input and output neurons, of a layer

        self.base_init = tf.truncated_normal_initializer(stddev=0.000001)#tf.contrib.layers.variance_scaling_initializer()#tf.initializers.he_normal#tf.truncated_normal_initializer(stddev=0.001)  # Initialise weights 0.01 at max
        self.reg_init = tf.contrib.layers.l2_regularizer(
            scale=0.00001)  # Initialise regularisation Alexnet 0.01 // 0.001 seems to make more sense

        f =self.apply_filter(self.f,gauss_kernel,name='face')
        le = self.apply_filter(self.le,gauss_kernel,name='eye')
        #le = self.Unet(le)
        #le = self.Unet(le,name='U2')
        re = self.apply_filter(self.re,gauss_kernel,name='eye',reuse=True)
        #re = self.Unet(re, reuse=True)
        #re = self.Unet(re, name='U2',reuse=True)

        fa = tf.layers.conv2d(f, filters=96, kernel_size=11, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first', activation=tf.nn.relu)


        fa = tf.layers.batch_normalization(fa, training=self.training, axis=1)
        fa = tf.layers.conv2d(fa, filters=256, kernel_size=5, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first', activation=tf.nn.relu)
        fa = tf.layers.batch_normalization(fa, training=self.training, axis=1)
        fa = tf.layers.conv2d(fa, filters=384, kernel_size=3, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first', activation=tf.nn.relu)
        fa = tf.layers.batch_normalization(fa, training=self.training, axis=1)
        fa = tf.layers.conv2d(fa, filters=64, kernel_size=1, strides=1, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first', activation=tf.nn.relu)
        fa = tf.layers.batch_normalization(fa, training=self.training, axis=1)
        le = tf.layers.conv2d(le, filters=96, kernel_size=11, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first',name='conv_E1', activation=tf.nn.relu)


        le = tf.layers.batch_normalization(le, training=self.training, axis=1)
        re = tf.layers.conv2d(re, filters=96, kernel_size=11, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='same', data_format='channels_first',name='conv_E1',reuse=True, activation=tf.nn.relu)


        re = tf.layers.batch_normalization(re, training=self.training, axis=1)

        le = tf.layers.conv2d(le, filters=256, kernel_size=5, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='same', data_format='channels_first',name='conv_E2', activation=tf.nn.relu)
        le = tf.layers.batch_normalization(le, training=self.training, axis=1)
        re = tf.layers.conv2d(re, filters=256, kernel_size=5, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='same', data_format='channels_first',name='conv_E2',reuse=True, activation=tf.nn.relu)
        re = tf.layers.batch_normalization(re, training=self.training, axis=1)

        le = tf.layers.conv2d(le, filters=384, kernel_size=3, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first',name='conv_E3', activation=tf.nn.relu)
        le = tf.layers.batch_normalization(le, training=self.training, axis=1)
        re = tf.layers.conv2d(re, filters=384, kernel_size=3, strides=2, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first',name='conv_E3',reuse=True, activation=tf.nn.relu)
        re = tf.layers.batch_normalization(re, training=self.training, axis=1)

        le = tf.layers.conv2d(le, filters=64, kernel_size=1, strides=1, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first',name='conv_E4', activation=tf.nn.relu)
        le = tf.layers.batch_normalization(le, training=self.training, axis=1)
        re = tf.layers.conv2d(re, filters=64, kernel_size=1, strides=1, kernel_initializer=self.base_init,kernel_regularizer=self.reg_init,
                             padding='valid', data_format='channels_first',name='conv_E4',reuse=True, activation=tf.nn.relu)
        re = tf.layers.batch_normalization(re, training=self.training, axis=1)
        # Flatten the 50 feature maps down to one vector
        re = tf.contrib.layers.flatten(re)
        le = tf.contrib.layers.flatten(le)
        hex = tf.contrib.layers.flatten(self.h)
        f = tf.contrib.layers.flatten(fa)

        #er = tf.contrib.layers.flatten(er)
        # change impact factor of face with size

        x = tf.layers.dense(re, units=128, kernel_initializer=self.base_init, name='FC-RE1', activation=tf.nn.relu,kernel_regularizer=self.reg_init)
        x = tf.layers.batch_normalization(x, training=self.training, axis=1)

        y = tf.layers.dense(le, units=128, kernel_initializer=self.base_init, name='FC-LE1', activation=tf.nn.relu,kernel_regularizer=self.reg_init)
        y = tf.layers.batch_normalization(y, training=self.training, axis=1)
        hex = tf.layers.dense(hex, units=64, kernel_initializer=self.base_init, name='FC-F2', activation=tf.nn.relu)
        hex = tf.layers.batch_normalization(hex, training=self.training, axis=1)

        z = tf.layers.dense(f, units=256, activation=tf.nn.relu, name='F-FC1',kernel_regularizer=self.reg_init)
        z = tf.layers.batch_normalization(z, training=self.training, axis=1)
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu, name='F-FC2',kernel_regularizer=self.reg_init)
        z = tf.layers.batch_normalization(z, training=self.training, axis=1)
        #z = tf.layers.dense(er, units=64, activation=tf.nn.relu, name='ER-FC1')

        s = tf.concat([x, y, z, hex ], 1)
        x = tf.layers.dense(s, units=128, name='FC0', kernel_regularizer=self.reg_init, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=64, name='FC1',kernel_regularizer=self.reg_init)
        x = tf.layers.dense(x, units=1, name='FC2')
        y = tf.layers.dense(s, units=128, name='sFC0', kernel_regularizer=self.reg_init, activation=tf.nn.relu)
        y = tf.layers.dense(y, units=64, name='sFC1',kernel_regularizer=self.reg_init)
        y = tf.layers.dense(y, units=1, name='sFC2')
        self.predictions = tf.concat([x,y],1)#*(pi/2) #pred


        g = self.g#*2/pi

        #pitch is y,yaw is x

        self.loss = tf.losses.mean_squared_error(self.g, self.predictions)+tf.losses.get_regularization_loss()#self.g
        self.angular_loss = util.gaze.tensorflow_angular_error_from_pitchyaw(self.g, self.predictions)#self.g
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.extra_update_ops = tf.get_collection(
            tf.GraphKeys. UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = self.trainer.minimize(self.loss)

        #TODO try with dropout!

