import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
path='../../data/assign3/connect-4.data'
f=open(path,'r')
f=f.read()

def adddata(stri,data,label):
	if stri=='':
		return [data,label]
	stri=stri.split(',')
	z=[0 for i in range(42*3)]
	for j in range(1,43):
		if(stri[j-1]=='o'):
			z[3*j-1]=1
		elif(stri[j-1]=='b'):
			z[3*j-2]=1
		else:
			z[3*j-3]=1
	a=0
	data.append(z)
	if(stri[42]=="win"):
		a=1
	elif(stri[42]=="loss"):
		a=-1
	label.append(a)
	return [data,label]

def data(index):
	data=[]
	label=[]
	j=len(f.split('\n'))-2
	if(index!=0):
		for i in f.split('\n')[0:index*j/5]:
			[data,label]=adddata(i,data,label)
	if(index!=4):
		for i in f.split('\n')[(index+1)*j/5:]:
			[data,label]=adddata(i,data,label)
	return [data,label]


def labeldata(index):
	data=[]
	label=[]
	j=len(f.split('\n'))-2
	if(index==4):
		y=j
	else:
		y=(index+1)*j/5
	for i in f.split('\n')[index*j/5:y]:
		[data,label]=adddata(i,data,label)
	return [data,label]

for i in range(0,5):
	[traindata,trainlabel]=data(i)
	[testdata,testlabel]=labeldata(i)
	print 'Batch number %d'%(i+1)
	print "One versus One"
	clf= SVC()
	clf.fit(traindata,trainlabel)
	prediction=clf.predict(testdata)
	print "Accuracy is:"
	print '%.4f'%(metrics.accuracy_score(prediction,testlabel))
	print 'One versus All'
	clf = LinearSVC()
	clf.fit(traindata,trainlabel)
	prediction=clf.predict(testdata)
	print 'Accuracy is:'
	print '%.4f'%(metrics.accuracy_score(prediction,testlabel))
		