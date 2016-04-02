import os
import sys
import struct
import numpy as np
from sklearn import metrics
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
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
for k in range(10000):
    temp = hog(img[k].reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    train_data.append(temp)
# # print rows,cols
for i in range(10000):
    label.append(lbl[i])

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
for k in range(1000):
    temp = hog(img[k].reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    test_data.append(temp)

for i in range(1000):
    test_label.append(lbl[i])

x=[]
y=[]
for i in range(1,300,2):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(train_data,label)
    prediction=clf.predict(test_data)
    temp=metrics.accuracy_score(prediction,test_label)
    print i,1-temp
    x.append(i)
    y.append(1-temp)
	# print(metrics.classification_report(test_label,prediction))
# for i in range(577,867):
#     print prediction[i-577],label[i]
plt.plot(x, y, linewidth=2.0)
plt.show()