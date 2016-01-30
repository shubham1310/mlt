import os
import struct
import numpy as np

pat='/home/shubh/sem6/MLT/Assignments/data/assign2/mnist/'
def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

# def show(image):
#     from matplotlib import pyplot
#     import matplotlib as mpl
#     fig = pyplot.figure()
#     ax = fig.add_subplot(1,1,1)
#     imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
#     imgplot.set_interpolation('nearest')
#     ax.xaxis.set_ticks_position('top')
#     ax.yaxis.set_ticks_position('left')
#     pyplot.show()

a=read("training",pat)

# for x in a:
