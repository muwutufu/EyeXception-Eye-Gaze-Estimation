#!/usr/bin/env python3

import tensorflow as tf
import parameters
from models import GazeNetPlus,NMOD
import numpy as np
from tqdm import tqdm, trange #progressbar



def TFRparsert(example): # Test file parser
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


def TFRecord2FLRD(filenames, batchsize=1):  # Parsing test files
    train_dataset = tf.data.TFRecordDataset(filenames=[filenames])
    train_dataset = train_dataset.map(TFRparsert)
    train_dataset = train_dataset.batch(batchsize)
    return train_dataset.make_initializable_iterator()


ditert = TFRecord2FLRD(filenames=parameters.PATHTE)
erdata, fdata, fldata, hdata, ledata, redata = ditert.get_next()


npx = np.empty([12500, 2])
nvnet = GazeNetPlus()
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(
device_count={'GPU': 1})
with tf.Session(config=config) as sess:
    # Restore the model
    print('Loading saved model...')
    print('Loading from: ', parameters.SAVE_PATH + parameters.MODEL_NAME + '.meta')
    restorer = tf.train.import_meta_graph(parameters.SAVE_PATH + parameters.MODEL_NAME + '.meta')
    restorer.restore(sess, tf.train.latest_checkpoint(parameters.SAVE_PATH))
    print("Model successfully restored")
    sess.run(ditert.initializer)
    it = 0
    try:
        with tqdm(desc='Iterations', leave=False) as pbar:
            while True:
                er, f, fl, h, le, re = sess.run([erdata, fdata, fldata, hdata, ledata, redata])
                train_dict = {
                    nvnet.training: False,
                    nvnet.er: er,
                    nvnet.f: f,
                    nvnet.fl: fl,
                    nvnet.h: h,
                    nvnet.le: le,
                    nvnet.re: re,
                }
                pred = sess.run(nvnet.predictions, feed_dict=train_dict)
                npx[it,:] = pred
                it += 1
                pbar.update()
    except tf.errors.OutOfRangeError:
        pass
    output_file = '%s/predictions.txt.gz' % (parameters.SAVE_PATH)
    np.savetxt(output_file, npx)#np.array

