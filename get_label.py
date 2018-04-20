from scipy.io import loadmat
import pandas as pd
import numpy as np

meta = loadmat('devkit/cars_meta.mat')

labels = list()
for l in meta['class_names'][0]:
    labels.append(l[0])
    print(l[0])