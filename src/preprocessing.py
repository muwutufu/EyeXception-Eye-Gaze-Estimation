#!/usr/bin/env python3

###################################################################
# !!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# DO NOT USE IF YOU DON'T HAVE ENOUGH RAM AND MEMORY SPACE ON DISK#
# EXTREMELY SLOW ANYWAY, NOT OPTIMIZED, NOT PARALLELIZED          #
###################################################################

import h5py
import numpy as np
from tempfile import mkdtemp
import os.path as pathu
import tensorflow as tf

# TODO: insert the path of where your h5 files are here
path = '/media/peter/E410A0F210A0CCBC/MPPROJDATA/old/'


def Npy2TFRecord(eyereg, face, faceland, gaze, head, lefteye, righteye, OutputPath, Size):  # , Size = num_entries
    writer = tf.python_io.TFRecordWriter(OutputPath)
    for h in range(Size):
        features = {
            'eye-region': tf.train.Feature(float_list=tf.train.FloatList(value=eyereg[h, :, :, :].flatten())),
            'face': tf.train.Feature(float_list=tf.train.FloatList(value=face[h, :, :, :].flatten())),
            'face-landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=faceland[h, :, :].flatten())),
            'head': tf.train.Feature(float_list=tf.train.FloatList(value=head[h, :].flatten())),
            'left-eye': tf.train.Feature(float_list=tf.train.FloatList(value=lefteye[h, :, :, :].flatten())),
            'right-eye': tf.train.Feature(float_list=tf.train.FloatList(value=righteye[h, :, :, :].flatten())),
            'gaze': tf.train.Feature(float_list=tf.train.FloatList(value=gaze[h, :].flatten()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()


def Npy2TFRecordt(eyereg, face, faceland, head, lefteye, righteye, OutputPath, Size):  # , Size = num_entries
    writer = tf.python_io.TFRecordWriter(OutputPath)
    for h in range(Size):
        features = {
            'eye-region': tf.train.Feature(float_list=tf.train.FloatList(value=eyereg[h, :, :, :].flatten())),
            'face': tf.train.Feature(float_list=tf.train.FloatList(value=face[h, :, :, :].flatten())),
            'face-landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=faceland[h, :, :].flatten())),
            'head': tf.train.Feature(float_list=tf.train.FloatList(value=head[h, :].flatten())),
            'left-eye': tf.train.Feature(float_list=tf.train.FloatList(value=lefteye[h, :, :, :].flatten())),
            'right-eye': tf.train.Feature(float_list=tf.train.FloatList(value=righteye[h, :, :, :].flatten())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()


def h52npy(hdf5, fn):
    keys_to_use = list(hdf5.keys())  # alternatively a list of keys that we select to use
    index_counter = 0
    index_to_key = {}
    for key in keys_to_use:
        n = next(iter(hdf5[key].values())).shape[0]
        for i in range(n):
            index_to_key[index_counter] = (key, i)
            index_counter += 1
    num_entries = index_counter
    entries_to_use = list(next(iter(hdf5.values())).keys())  # or list of string of entries we want to use
    current_index = 0
    print(num_entries)

    # initialize np array for faster performance
    entries = {}
    for name in entries_to_use:
        ex_dim = hdf5[keys_to_use[0]][name].shape
        # entries[name] = np.empty([num_entries,*ex_dim[1:]])
        if len(ex_dim) == 4:
            entries[name] = np.memmap(fn + name + '.dat', dtype='float32', mode='w+',
                                      shape=(num_entries, ex_dim[-1], *ex_dim[1:-1]))
        else:
            entries[name] = np.memmap(fn + name + '.dat', dtype='float32', mode='w+', shape=(num_entries, *ex_dim[1:]))

    for i in range(num_entries):
        key, index = index_to_key[i]
        data = hdf5[key]
        for name in entries_to_use:
            if data[name][index].ndim == 3:
                touse = np.transpose(data[name][index], [2, 0, 1])
            else:
                touse = data[name][index]
            entries[name][i] = touse
    return entries, num_entries, entries_to_use


ne = np.empty(3)
etu = {}
fn1 = pathu.join(mkdtemp(), 'newfile1')
hdf5 = h5py.File(path + 'mp19_train.h5', 'r')
print('train_size:')
train, ne[0], etu[0] = h52npy(hdf5, fn1)

fn2 = pathu.join(mkdtemp(), 'newfile2')
hdf5 = h5py.File(path + 'mp19_validation.h5', 'r')
print('val_size:')
val, ne[1], etu[1] = h52npy(hdf5, fn2)

fn3 = pathu.join(mkdtemp(), 'newfile3')
hdf5 = h5py.File(path + 'mp19_test_students.h5', 'r')
print('test_size:')
test, ne[2], etu[2] = h52npy(hdf5, fn3)

# standardization over each color channel (using training set only!!), instead of normalization
# train
for en in etu[0]:
    if train[en].ndim == 4:
        m = np.empty(3)
        s = np.empty(3)
        for j in range(3):
            m[j] = np.mean(train[en][:, j, :, :])
            train[en].flush()
            s[j] = np.std(train[en][:, j, :, :])
            train[en].flush()
# val and test
for i, t in enumerate([train, val, test]):
    for en in etu[i]:
        if t[en].ndim == 4:
            for j in range(3):
                if i == 0 and en != 'eye-region':
                    for c in range(int(ne[i])):
                        t[en][c, j, :, :] = (t[en][c, j, :, :] - m[j]) / s[j]
                        if c % 10000 == 0:
                            t[en].flush()
                    t[en].flush()
                else:
                    t[en][:, j, :, :] = (t[en][:, j, :, :] - m[j]) / s[j]
                    t[en].flush()
                print('done looping with ' + str(i) + ':' + str(j) + ' ' + en)
for i, t in enumerate([train, val]):
    Npy2TFRecord(t['eye-region'], t['face'], t['face-landmarks'], t['gaze'], t['head'], t['left-eye'],
                 t['right-eye'], path + 'DATA' + str(i), int(ne[i]))
    for en in etu[i]:
        t[en].flush()

Npy2TFRecordt(test['eye-region'], test['face'], test['face-landmarks'], test['head'], test['left-eye'],
              test['right-eye'], path + 'DATA' + str(2), int(ne[2]))