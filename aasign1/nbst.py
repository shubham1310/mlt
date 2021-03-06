import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
#path = sys.argv[1]
path='../../data/assign2/bare/'
folder=os.listdir(path)
lmtzr = WordNetLemmatizer()
data=[]
label=[]

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
                lem=[porter_stemmer.stem(word) for word in text.split(" ")]
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
                #Comment the next two lines if you don't want lemmatization
                lem=[porter_stemmer.stem(word) for word in text.split(" ")]
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
        # print text
        #Comment the next two lines if you don't want lemmatization
        lem=[porter_stemmer.stem(word) for word in text.split(" ")]
        text = " ".join([str(sentence) for sentence in lem])
        # print documents
        data.append(text)
    # print len(label)

#remove stop_words = "english" if you don't want them to be removed
count_vect = CountVectorizer(stop_words = "english")#stop_words = "english"
train_data = count_vect.fit_transform(data)


# train_data = train_data.toarray()
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
    traindata=data[:size]
    trainlabel=label[:size]
    testdata = data[size-1:]
    testlabel = label[size-1:]
    clf=MultinomialNB()
    clf.fit(traindata,trainlabel)
    prediction=clf.predict(testdata)
    print 'Taking part %d as the test data the accuracy is: '%(i),
    print '%.4f'%(metrics.accuracy_score(prediction,testlabel))
    #print(metrics.classification_report(testlabel,prediction))
