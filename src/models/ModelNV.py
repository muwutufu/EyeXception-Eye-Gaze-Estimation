#!/usr/bin/env python3

import tensorflow as tf
import util.gaze
import parameters
import numpy as np



basefilt = 16
basefiltf = 32
base_filt = 16
class Model1NetNV():
    """An example neural network architecture."""

    def conv_batch_relu(self, tensor, filters, kernel=[3, 3], stride=[1, 1], is_training=True, name='convbr',reuse=None):
        conv = tf.layers.conv2d(tensor, filters, kernel_size=kernel, strides=stride, padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first", name=name,reuse=reuse)
        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        conv = tf.nn.relu(conv)
        return conv

    # def centre_crop_and_concat(self, prev_conv, up_conv): #from https://github.com/96imranahmed/3D-Unet
    #     # If concatenating two different sized Tensors, centre crop the first Tensor to the right size and concat
    #     # Needed if you don't have padding
    #     p_c_s = prev_conv.get_shape()
    #     u_c_s = up_conv.get_shape()
    #     offsets = np.array([0, 0, (p_c_s[2] - u_c_s[2]) // 2, (p_c_s[3] - u_c_s[3]) // 2,
    #                         (p_c_s[4] - u_c_s[4]) // 2], dtype=np.int32)
    #     size = np.array([-1, p_c_s[1], u_c_s[2], u_c_s[3], u_c_s[4]], np.int32)
    #     prev_conv_crop = tf.slice(prev_conv, offsets, size)
    #     up_concat = tf.concat((prev_conv_crop, up_conv), 1)
    #     return up_concat




    def conv3by3(self,tensor,name, filters):

        conv3 = tf.layers.conv2d(tensor, filters, kernel_size=[1, 3], strides=[1, 1], padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")

        conv3 = tf.layers.conv2d(conv3, filters, kernel_size=[3, 1], strides=[1, 1], padding='same',
                                 kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                 data_format="channels_first")

        return conv3

    def inception_module(self,tensor,name, filters, is_training=True):

        conv = tf.layers.conv2d(tensor, filters, kernel_size=[1,1], strides=[1,1], padding='same',
                            kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                            data_format='channels_first')
        self.summary.filters('filters', conv)
        self.summary.feature_maps('features', conv, data_format='channels_first')

        conv3 = self.conv3by3(conv,name+'_conv3_', filters)

        conv5 = self.conv3by3(conv,name+'_conv5_1_', filters)
        conv5 = self.conv3by3(conv5,name+'_conv5_2_', filters)

        conv7 = self.conv3by3(conv,name+'_conv7_1_', filters)
        conv7 = self.conv3by3(conv7,name+'_conv7_2_', filters)
        conv7 = self.conv3by3(conv7,name+'_conv7_3_', filters)

        conv = tf.concat([conv, conv3, conv5, conv7],1)
        #conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
        conv = tf.nn.relu(conv)
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
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv4 = tf.layers.separable_conv2d(conv3, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
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
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
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
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end

    def Middle_flow(self, tensor, name, reuse=None, is_training=True):

        m1 = tf.nn.relu(tensor)
        conv1 = tf.layers.separable_conv2d(m1, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conv1', reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)
        conv2 = tf.layers.separable_conv2d(conv1, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conv2', reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                depthwise_initializer=self.base_init,pointwise_initializer=self.base_init,
                                depthwise_regularizer=self.reg_init,pointwise_regularizer=self.reg_init,
                                           data_format="channels_first", name=name + '_conv3', reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3 = tf.concat([conv3,tensor],1)
        return conv3













    def __init__(self, learning_rate=parameters.LEARNING_RATE):
        self.f = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
        self.er = tf.placeholder(tf.float32, shape=(None, 3, 60, 224))
        self.le = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.re = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.h = tf.placeholder(tf.float32, shape=(None, 2))
        self.fl = tf.placeholder(tf.float32, shape=(None, 33, 2))
        self.g = tf.placeholder(tf.float32, shape=(None, 2))
        self.LR = tf.placeholder(tf.float32)

        self.drop = 0.5
        self.training = tf.placeholder(tf.bool)

        self.base_init = tf.truncated_normal_initializer(stddev=0.1)  # Initialise weights
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.0001)  # Initialise regularisation


        #BEGIN UX
        re = self.Unet(self.re,name='eu')
        le = self.Unet(self.le, name='eu',reuse=True)#, reuse=True)
        re = self.xception_e(re,'xe',is_training=self.training)
        le = self.xception_e(le,'xe',reuse=True,is_training=self.training)
        x = tf.contrib.layers.flatten(re)
        y = tf.contrib.layers.flatten(le)
        x = tf.concat([x,y,self.h],1)#add tf,atan(self.h)
        #END UX


        #hf1 = tf.layers.dense(self.h, units=2, name='hf1')



        x = tf.layers.dense(x, units=2, name='FC2')#, activation=tf.nn.relu


        self.predictions = x#tf.atan(x)

        #pitch is y,yaw is x

        self.loss = tf.losses.mean_squared_error(self.g, x)#+tf.losses.get_regularization_loss()
        self.angular_loss = util.gaze.tensorflow_angular_error_from_pitchyaw(self.g, x)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.LR)
        self.extra_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = self.trainer.minimize(self.loss)

