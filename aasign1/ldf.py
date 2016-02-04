import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from nltk.stem.wordnet import WordNetLemmatizer
#path = sys.argv[1]
path='../../data/assign2/bare/'
folder=os.listdir(path)
data=[]
label=[]
lmtzr = WordNetLemmatizer()

def ttdata(i):
    data=[]
    label=[]
    size=0
    for x in folder:
        npath=os.path.join(path,x)
        sfolder=os.listdir(npath)
        if(int(x[-1])!=(i%10)):
            for y in sfolder:
                if(y[:3]=='spm'):
                    label.append('1')
                else:
                    label.append('0')
                f=open(os.path.join(npath,y),'r')
                text=f.read()
                # print text
                lem=[lmtzr.lemmatize(word) for word in text.split(" ")]
                text = " ".join([str(sentence) for sentence in lem])
                data.append(text)
    size=len(label)
    for x in folder:
        npath=os.path.join(path,x)
        sfolder=os.listdir(npath)
        if(int(x[-1])==(i%10)):
            # print x
            for y in sfolder:
                if(y[:3]=='spm'):
                    label.append('1')
                else:
                    label.append('0')
                f=open(os.path.join(npath,y),'r')
                text=f.read()
                # print text
                lem=[lmtzr.lemmatize(word) for word in text.split(" ")]
                text = " ".join([str(sentence) for sentence in lem])
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
        lem=[lmtzr.lemmatize(word) for word in text.split(" ")]
        text = " ".join([str(sentence) for sentence in lem])
        data.append(text)
    # print len(label)
#Comment the next line if you want other than binary representation
# count_vect = CountVectorizer(binary='True',stop_words = "english")

#uncomment the next line for other than binary representation
count_vect = CountVectorizer(stop_words = "english")

train_data = count_vect.fit_transform(data)

#uncomment the next two lines for term frequency
train_data = train_data.toarray()
count_vect2 = TfidfTransformer(use_idf=False,sublinear_tf=True)

#uncomment the next line for tf-idf BoW representation
# count_vect2 = TfidfTransformer()
train_data = count_vect2.fit_transform(train_data)
train_data = train_data.toarray()
# for x in train_data[0]:
#     print x
# print train_data[0]
# print train_data.shape
# vocab=count_vect.get_feature_names()
#print vocab
# dist = np.sum(train_data, axis=0)
# for tag, count in zip(vocab, dist):
#     print count, tag

for i in range(1,11):
    [data,label,size] = ttdata(i)
    data = count_vect.fit_transform(data)
    data = data.toarray()
    data = count_vect2.fit_transform(data)
    data = data.toarray()
    traindata=data[:size]
    trainlabel=label[:size]
    testdata = data[size-1:]
    testlabel = label[size-1:]
    clf=linear_model.Perceptron(alpha =0,fit_intercept=True)
    clf.fit(traindata,trainlabel)
    prediction=clf.predict(testdata)
    print 'Taking part %d as the test data the accuracy is:'%(i),
    print '%.4f'%(metrics.accuracy_score(prediction,testlabel))
    # print(metrics.classification_report(testlabel,prediction))
# for i in range(577,867):
#     print prediction[i-577],label[i]
