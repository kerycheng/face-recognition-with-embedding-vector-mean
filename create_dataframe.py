import warnings
warnings.filterwarnings('ignore')

import os
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True)

import pandas as pd
import scipy.misc
import itertools
import random
import cv2
import facenet
import detect_face

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import visualization_utils as vis_utils
from sklearn.cluster import MeanShift, estimate_bandwidth

from basic_settings import setup_settings
from basic_settings import calculation

class create_dataframe(setup_settings, calculation):
    def __init__(self):
        super().__init__()



