import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#path = sys.argv[1]
path='../../data/assign2/bare/'
folder=os.listdir(path)
data=[]
label=[]

def ttdata(i):
    data=[]
    label=[]
    size=0
    print 'train'
    for x in folder:
        npath=os.path.join(path,x)
        sfolder=os.listdir(npath)
        if(int(x[-1])%5!=i):
            print x
            for y in sfolder:
                if(y[:3]=='spm'):
                    label.append('1')
                else:
                    label.append('0')
                f=open(os.path.join(npath,y),'r')
                text=f.read()
                data.append(text)
    size=len(label)
    print 'test'
    for x in folder:
        npath=os.path.join(path,x)
        sfolder=os.listdir(npath)
        if(int(x[-1])%5==i):
            print x
            for y in sfolder:
                if(y[:3]=='spm'):
                    label.append('1')
                else:
                    label.append('0')
                f=open(os.path.join(npath,y),'r')
                text=f.read()
                data.append(text)
    return [data,label,size]


for x in folder:
    npath=os.path.join(path,x)
    sfolder=os.listdir(npath)
    for y in sfolder:
        if(y[:3]=='spm'):
            label.append('1')
        else:
            label.append('0')
        f=open(os.path.join(npath,y),'r')
        text=f.read()
        data.append(text)
    # print len(label)

#remove stop_words = "english" if you don't want them to be removed
count_vect = CountVectorizer(stop_words = "english")
train_data = count_vect.fit_transform(data)


for i in range(0,5):
    [data,label,size] = ttdata(i)
    data = count_vect.fit_transform(data)
    data = data.toarray()
    traindata=data[:size]
    trainlabel=label[:size]
    testdata = data[size-1:]
    testlabel = label[size-1:]
    clf = SVC()
    clf.fit(traindata,trainlabel)
    prediction=clf.predict(testdata)
    print 'Standard formulation'
    print 'Batch number %d as the test data the accuracy is: '%(i),
    print '%.4f'%(metrics.accuracy_score(prediction,testlabel))
    #print(metrics.classification_report(testlabel,prediction))
    clf = LinearSVC(loss='hinge')
    clf.fit(traindata,trainlabel)
    prediction=clf.predict(testdata)
    print 'Hinge Loss'
    print 'Batch number %d  as the test data the accuracy is: '%(i),
    print '%.4f'%(metrics.accuracy_score(prediction,testlabel))