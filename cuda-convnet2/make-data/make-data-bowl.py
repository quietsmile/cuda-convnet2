# Copyright 2014 Google Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################


# This script makes batches suitable for training from raw ILSVRC 2012 tar files.

import tarfile
from StringIO import StringIO
from random import shuffle
import sys
from time import time
from pyext._MakeDataPyExt import resizeJPEG
import itertools
import os
import cPickle
import scipy.io
import math
import argparse as argp
import numpy as np
from PIL import Image

# Set this to True to crop images to square. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels, and then the
# center OUTPUT_IMAGE_SIZE x OUTPUT_IMAGE_SIZE patch will be extracted.
#
# Set this to False to preserve image borders. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels. This was
# demonstrated to be superior by Andrew Howard in his very nice paper:
# http://arxiv.org/abs/1312.5402
CROP_TO_SQUARE          = True
OUTPUT_IMAGE_SIZE       = 256

# Number of threads to use for JPEG decompression and image resizing.
NUM_WORKER_THREADS      = 8

# Don't worry about these.
OUTPUT_BATCH_SIZE = 3072
OUTPUT_SUB_BATCH_SIZE = 1024

def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)

def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents

def partition_list(l, partition_size):
    divup = lambda a,b: (a + b - 1) / b
    return [l[i*partition_size:(i+1)*partition_size] for i in xrange(divup(len(l),partition_size))]

def open_tar(path, name):
    if not os.path.exists(path):
        print "ILSVRC 2012 %s not found at %s. Make sure to set ILSVRC_SRC_DIR correctly at the top of this file (%s)." % (name, path, sys.argv[0])
        sys.exit(1)
    return tarfile.open(path)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_devkit_meta(ILSVRC_DEVKIT_TAR):
    tf = open_tar(ILSVRC_DEVKIT_TAR, 'devkit tar')
    fmeta = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/meta.mat'))
    meta_mat = scipy.io.loadmat(StringIO(fmeta.read()))
    labels_dic = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names_dic = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]

    fval_ground_truth = tf.extractfile(tf.getmember('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
    tf.close()
    return labels_dic, label_names, validation_ground_truth


def write_batches(target_dir, name, start_batch_num, labels, jpeg_files):
    jpeg_files = partition_list(jpeg_files, OUTPUT_BATCH_SIZE)
    labels = partition_list(labels, OUTPUT_BATCH_SIZE)
    makedir(target_dir)
    print "Writing %s batches..." % name
    for i,(labels_batch, jpeg_file_batch) in enumerate(zip(labels, jpeg_files)):
        t = time()
        jpeg_strings = list(itertools.chain.from_iterable(resizeJPEG([jpeg.read() for jpeg in jpeg_file_batch], OUTPUT_IMAGE_SIZE, NUM_WORKER_THREADS, CROP_TO_SQUARE)))
        batch_path = os.path.join(target_dir, 'data_batch_%d' % (start_batch_num + i))
        makedir(batch_path)
        for j in xrange(0, len(labels_batch), OUTPUT_SUB_BATCH_SIZE):
            pickle(os.path.join(batch_path, 'data_batch_%d.%d' % (start_batch_num + i, j/OUTPUT_SUB_BATCH_SIZE)), 
                   {'data': jpeg_strings[j:j+OUTPUT_SUB_BATCH_SIZE],
                    'labels': labels_batch[j:j+OUTPUT_SUB_BATCH_SIZE]})
        print "Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i+1, len(jpeg_files), time() - t)
    return i + 1

if __name__ == "__main__":
    prefix = '../../bowl_data/data/'
    train_path = prefix + 'train.tar'
    test_path = prefix + 'test.tar'
    target_path = '../../bowl_data/batches/'

    labels = open('../../bowl_data/labels.txt').readlines()[0].strip().split()
    labels_dic = {}
    cnt = 0
    for i in labels:
        labels_dic[i] = cnt
        cnt += 1

    train_jpeg_files = []
    train_labels = []
    with open_tar(train_path, 'training tar') as tf:
        members = [ i for i in tf.getmembers() if len(i.name.strip().split('/')) == 3]
        names = tf.getnames()
        train_jpeg_files = [ tf.extractfile(m) for m in members ]
        file_classnames = [ m.name.split('/')[-2] for m in members ]

        train_idx = range(len(train_jpeg_files))
        shuffle(train_idx)

        train_list = [train_jpeg_files[i] for i in train_idx[:3072*9]]
        train_labels = [[labels_dic[file_classnames[i]]] for i in train_idx[:3072*9]]
        val_list = [train_jpeg_files[i] for i in train_idx[3072*9:]]
        val_labels = [[labels_dic[file_classnames[i]]] for i in train_idx[3072*9:]]

        write_batches(target_path, 'training', 0, train_labels, train_list)
        write_batches(target_path, 'validation', 100, val_labels, val_list)

    with open_tar(test_path, 'testing tar') as tf:
        members = [ i for i in tf.getmembers() if len(i.name.strip().split('/')) == 2]
        print members[1:10]
        test_list = [ tf.extractfile(m) for m in members ]
        test_labels = [[0]]*len(test_list)
        print test_list[1:10]
        write_batches(target_path, 'test', 1000, test_labels, test_list)

    mean = Image.fromarray(unpickle('batches.meta')['data_mean'].reshape((128,128))).resize((OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    mean = np.array(mean.convert('RGB')).swapaxes(0,1).swapaxes(0,2).flatten()
    meta = unpickle('input_meta')
    meta_file = os.path.join( target_path, 'batches.meta')
    meta.update({'batch_size': OUTPUT_BATCH_SIZE,
                 'num_vis': OUTPUT_IMAGE_SIZE ** 2 * 3,
                 'label_names': labels,
                 'data_mean': mean})
    pickle(meta_file, meta)

  


