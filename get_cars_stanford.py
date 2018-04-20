# encoding:utf8

from scipy.io import loadmat
import pandas as pd
import numpy as np
import sys

mat_train = loadmat('devkit/cars_train_annos.mat')
mat_test = loadmat('devkit/cars_test_annos.mat')
meta = loadmat('devkit/cars_meta.mat')

labels = list()
for l in meta['class_names'][0]:
    labels.append(l[0])

print(len(labels))
sys.exit()
    
train = list()
for example in mat_train['annotations'][0]:
    label = example[-2][0][0]-1
    image = example[-1][0]
    train.append((image,label))
    
test = list()
for example in mat_test['annotations'][0]:
    image = example[-1][0]
    test.append(image)

validation_size = int(len(train) * 0.10)
test_size = int(len(train) * 0.20)

validation = list(train[:validation_size])
np.random.shuffle(validation)
train = train[validation_size:]

test = list(train[:test_size])
np.random.shuffle(test)
train = train[test_size:]

# Google cloud example
bucket_path = ''
with open('cars_data.csv', 'w+') as f:
    [f.write('TRAIN,%s%s,%s\n' %(bucket_path,img,lab)) for img,lab in train]
    [f.write('VALIDATION,%s%s,%s\n' %(bucket_path,img,lab)) for img,lab in validation]
    [f.write('TEST,%s%s\n' %(bucket_path,img)) for img,_ in test]
