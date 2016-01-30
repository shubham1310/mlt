import os
import sys
import struct
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# path='/home/shubh/sem6/MLT/Assignments/data/assign2/mnist/'
path='../../data/assign2/mnist/'
# path = sys.argv[1]
train_data=[]
label=[]
fname_img = os.path.join(path, 'train-images.idx3-ubyte')
fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')

flbl = open(fname_lbl, 'rb')
magic, num = struct.unpack(">II", flbl.read(8))
lbl = np.fromfile(flbl, dtype=np.int8)

fimg= open(fname_img, 'rb')
magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
for k in range(len(lbl)):
    temp=[]
    for i in range(rows):
        for j in range(cols):
            # print i,j
            temp.append(img[k][i][j])
    train_data.append(temp)
# print train_data
# print rows,cols
for i in range((len(lbl))):
    # train_data.append(img[i])
    label.append(lbl[i])
print len(train_data)
print len(label)

test_data=[]
test_label=[]
fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
flbl = open(fname_lbl, 'rb')
magic, num = struct.unpack(">II", flbl.read(8))
lbl = np.fromfile(flbl, dtype=np.int8)

fimg = open(fname_img, 'rb')
magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
for k in range(len(lbl)):
    temp=[]
    for i in range(rows):
        for j in range(cols):
            # print i,j
            temp.append(img[k][i][j])
    test_data.append(temp)

for i in range((len(lbl))):
    # test_data.append(img[i])
    test_label.append(lbl[i])
print len(test_data)
print len(test_label)



# metric =  [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(train_data,label)
prediction=knn.predict(test_data)
print np.mean(prediction==test_label)
print metrics.accuracy_score(prediction,test_label)
print(metrics.classification_report(test_label,predictions))
# for i in range(577,867):
#     print prediction[i-577],label[i]
