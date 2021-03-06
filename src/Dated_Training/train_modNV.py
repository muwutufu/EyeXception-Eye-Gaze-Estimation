#!/usr/bin/env python3

import os
import tensorflow as tf
import parameters
from models import Model1NetNV
import numpy as np
from tqdm import tqdm, trange #progressbar

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #back to 0 afterwards
tf.logging.set_verbosity(tf.logging.ERROR)

def TFRparsert(example):#, Size=12500):
    features = {
        'eye-region': tf.FixedLenFeature([3, 60, 224], tf.float32),
        'face': tf.FixedLenFeature([3, 224, 224], tf.float32),
        'face-landmarks': tf.FixedLenFeature([33, 2], tf.float32),
        'head': tf.FixedLenFeature([2], tf.float32),
        'left-eye': tf.FixedLenFeature([3, 60, 90], tf.float32),
        'right-eye': tf.FixedLenFeature([3, 60, 90], tf.float32)
    }
    parsedf = tf.parse_single_example(example, features)
    return parsedf['eye-region'], parsedf['face'], parsedf['face-landmarks'], parsedf['head'], parsedf['left-eye'], parsedf['right-eye']
#val Size=4000):


def TFRparser(example):#, Size=100000):
    features = {
        'eye-region': tf.FixedLenFeature([3, 60, 224], tf.float32),
        'face': tf.FixedLenFeature([3, 224, 224], tf.float32),
        'face-landmarks': tf.FixedLenFeature([33, 2], tf.float32),
        'head': tf.FixedLenFeature([2], tf.float32),
        'left-eye': tf.FixedLenFeature([3, 60, 90], tf.float32),
        'right-eye': tf.FixedLenFeature([3, 60, 90], tf.float32),
        'gaze': tf.FixedLenFeature([2], tf.float32)
    }
    parsedf = tf.parse_single_example(example, features)
    return parsedf['eye-region'], parsedf['face'], parsedf['face-landmarks'], parsedf['head'], parsedf['left-eye'], parsedf['right-eye'], parsedf['gaze']

def TFRecord2FLRD(filenames, buffersize=3000, batchsize=parameters.BATCH_SIZE):#TODO set shufflesize
    train_dataset = tf.data.TFRecordDataset(filenames=[filenames])
    train_dataset = train_dataset.map(TFRparser)
    train_dataset = train_dataset.shuffle(buffersize)
    #train_dataset = train_dataset.prefetch(buffersize) #15 was too much#TODO: set prefetchsize
    train_dataset = train_dataset.batch(batchsize)
    return train_dataset.make_initializable_iterator()

tf.reset_default_graph()#tf.compat.v1.reset_default_graph

ditert = TFRecord2FLRD(filenames=parameters.PATHT)
diterv = TFRecord2FLRD(filenames=parameters.PATHV)
erdata, fdata, fldata, hdata, ledata, redata, gdata = ditert.get_next()  # here xdata is the input, and ydata what it will be compared to
erdatav, fdatav, fldatav, hdatav, ledatav, redatav, gdatav = diterv.get_next()



LR = parameters.LEARNING_RATE
nvnet = Model1NetNV()
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(
    device_count={'GPU': 1})  # XLA_GPU is experimental, might get errors, only ~10% better performance on ResNet50
with tf.Session(config=config) as sess:
    TBwriter = tf.summary.FileWriter(parameters.LOGS_PATH, sess.graph) #TODO: currently saving values at beginning of iterations
    bestvalloss = float('inf')
    trloss = tf.summary.scalar('mse_training_loss', nvnet.loss)
    ltrloss = tf.summary.scalar('angular_training_loss', nvnet.angular_loss)
    valoss = tf.summary.scalar('mse_validation_loss', nvnet.loss)
    lvaloss = tf.summary.scalar('angular_validation_loss', nvnet.angular_loss)
    if parameters.LOAD_MODEL:
        print('Trying to load saved model...')
        try:
            print('Loading from: ', parameters.SAVE_PATH + parameters.MODEL_NAME + '.meta')
            restorer = tf.train.import_meta_graph(parameters.SAVE_PATH + parameters.MODEL_NAME + '.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(parameters.SAVE_PATH))
            print("Model successfully restored")
        except IOError:
            sess.run(init)
            print("No previous model found, running default initialization")
    v_loss = np.empty(parameters.EPOCHS)
    patience_c = 0
    for epoch_no in trange(parameters.EPOCHS, desc='Epochs', position=0):
        train_loss = 0
        train_aloss = 0
        val_loss = 0
        val_aloss = 0
        tr_loss = 0
        va_loss = 0
        sess.run(ditert.initializer)
        # x, y = sess.run([xdata, ydata])#xdata.eval(), ydata.eval()#sess.run([xdata, ydata])
        # # Initialize iterator with training data
        # train_dict = {
        #     unet.training: True,
        #     unet.model_input: x,#xdata,
        #     unet.model_cout: y#ydata
        # }
        itt = 0
        if epoch_no > 0 and epoch_no%4==0:
            LR *= 0.1
        try:
            with tqdm(total=int(100000/parameters.BATCH_SIZE)+1 ,desc='Batches', leave=False) as pbar:
                while True:
                    er, f, fl, h, le, re, g = sess.run([erdata, fdata, fldata, hdata, ledata, redata, gdata])  # xdata.eval(), ydata.eval()#sess.run([xdata, ydata])
                    # Initialize iterator with training data
                    #merge = tf.summary.merge_all()
                    train_dict = {
                        nvnet.training: True,
                        nvnet.LR: LR,
                        nvnet.er: er,
                        nvnet.f: f,
                        nvnet.fl: fl,
                        nvnet.h: h,
                        nvnet.le: le,
                        nvnet.re: re,
                        nvnet.g: g,
                    }
                    if itt == 0:
                        _, loss, aloss, traloss, ltraloss = sess.run([nvnet.train_op, nvnet.loss, nvnet.angular_loss, trloss, ltrloss], feed_dict=train_dict)
                        TBwriter.add_summary(traloss, (epoch_no+1))
                        TBwriter.add_summary(ltraloss, (epoch_no+1))
                    else:
                        _, loss, aloss = sess.run([nvnet.train_op, nvnet.loss, nvnet.angular_loss], feed_dict=train_dict)
                    train_loss += loss
                    train_aloss += aloss
                    itt += 1

                    pbar.set_postfix(MSE_Loss=loss, ANGULAR_Loss=aloss)
                    pbar.update()
        except tf.errors.OutOfRangeError:
            pass
        tr_loss = loss
        sess.run(diterv.initializer)

        itc = 0
        try:
            while True:
                er, f, fl, h, le, re, g = sess.run([erdatav, fdatav, fldatav, hdatav, ledatav, redatav, gdatav])  # xdatav.eval(), ydatav.eval()#sess.run([xdatav, ydatav])
                # Initialize iterator with validation data
                #merge = tf.summary.merge_all()
                train_dict = {
                    nvnet.training: False,
                    nvnet.er: er,
                    nvnet.f: f,
                    nvnet.fl: fl,
                    nvnet.h: h,
                    nvnet.le: le,
                    nvnet.re: re,
                    nvnet.g: g,
                }
                if itc == 0:
                    loss, aloss, vall, lvall = sess.run([nvnet.loss, nvnet.angular_loss, valoss, lvaloss], feed_dict=train_dict)
                    TBwriter.add_summary(vall, (epoch_no+1))
                    TBwriter.add_summary(lvall, (epoch_no+1))
                else:
                    loss, aloss = sess.run([nvnet.loss, nvnet.angular_loss], feed_dict=train_dict)
                val_loss += loss
                val_aloss += aloss
                itc += 1
        except tf.errors.OutOfRangeError:
            pass
        va_loss = loss
        tott_loss = train_loss / itt  # average training loss in 1 epoch #TODO maybe not get average training loss but only last
        totv_loss = val_loss / itc  # average validation loss in 1 epoch
        totat_loss = train_aloss / itt
        totav_loss = val_aloss / itc
        v_loss[epoch_no] = totv_loss  # average val loss saved for early stopping
        print('\nEpoch No: {}'.format(epoch_no + 1))
        print('MSE Train loss = {:.8f}'.format(tott_loss))
        print('Angular Train loss = {:.8f}'.format(totat_loss))
        print('MSE Val loss = {:.8f}'.format(totv_loss))
        print('Angular Val loss = {:.8f}'.format(totav_loss))

        if (bestvalloss - v_loss[epoch_no]) > 0.0000000001:
            print('Saving model at epoch: ', (epoch_no + 1))  # Save periodically
            saver.save(sess, parameters.SAVE_PATH + parameters.MODEL_NAME + 'best')
            bestvalloss = v_loss[epoch_no]
            patience_c = 0
        else:
            # can save model here if patience is too big
            patience_c += 1

        if patience_c > parameters.PATIENCE:
            print("early stopping...")
            break

        if (epoch_no+1) % 100 == 0 and epoch_no > 0:
            print('Saving model at epoch: ', (epoch_no + 1))  # Save periodically
            saver.save(sess, parameters.SAVE_PATH + parameters.MODEL_NAME, global_step=(epoch_no + 1))

    saver.save(sess, parameters.SAVE_PATH + parameters.MODEL_NAME, global_step=(
                epoch_no + 1))  # Final save #saver.save(sess, 'my-test-model', nr of setps after when to save)

    # closing session not needed with 'with'
