from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import urllib
import urllib2
import tarfile
import os
import random
import string
import cPickle
from Helper import Helper

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# for small is 124
# for large is 65466
vocabulary_size = 65466
basedir = os.getcwd()
data_index = 0
# Type your path of training data path
training_data_pth = "JakeDrive/training-data-large.txt"

target = os.path.join(basedir, 'all_data2.txt')
with open(os.path.join(target), "w") as fo:
  training_set = training_data_pth
  train_df = pd.read_csv(training_set, sep='\t', names=["Label", "Text"])
  train_text = train_df["Text"]
  for i in range(len(train_text)):
    fo.write(train_text[i])
    fo.write('\n')

