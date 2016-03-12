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

def adddata(stri,data1,data2,data3):
	stri=i.split(',')
	z=[0 for i in range(42*3)]
	for j in range(1,43):
		if(stri[j-1]=='o'):
			z[3*j-1]=1
		elif(stri[j-1]=='b'):
			z[3*j-2]=1
		else:
			z[3*j-3]=1
	a=0
	if(stri[42]=="win"):
		a=1
		data1.append(z)
	elif(stri[42]=="loss"):
		a=-1
		data2.append(z)
	else:
		data3.append(z)
	return [data1,data2,data3]

def data(index):
	data1=[]
	data2=[]
	data3=[]
	j=len(f.split('\n'))-2
	if(index!=0):
		for i in f.split('\n')[0:index*j/5]:
			[data1,data2,data3]=adddata(i,data1,data2,data3)
	if(index!=4):
		for i in f.split('\n')[(index+1)*j/5:]:
			[data1,data2,data3]=adddata(i,data1,data2,data3)
	return [data1,data2,data3]


def labeldata(index):
	data=[]
	label=[]
	j=len(f.split('\n'))-2
	if(j==4):
		y=j
	else:
		y=(index+1)*j/5
	for i in f.split('\n')[index*j/5:y]:
		x=i.split(',')
		z=[0 for i in range(42*3)]
		for j in range(1,43):
			if(x[j-1]=='o'):
				z[3*j-1]=1
			elif(x[j-1]=='b'):
				z[3*j-2]=1
			else:
				z[3*j-3]=1
		a=0
		data.append(z)
		if(x[42]=="win"):
			a=1
		elif(x[42]=="loss"):
			a=-1
		label.append[a]
	return [data,label]

for i in range(0,5):
	[data1,data2,data3]=data(i)
	[testdata,label]=labeldata(i)
	print 'Batch number %d'%(i)
	print "One versers One"
	clf1= SVC()
	clf1.fit(data1+data2,[1 for j in range(len(data1))]+[-1 for j in range(len(data2))])
	clf2= SVC()
	clf2.fit(data1+data3,[1 for j in range(len(data1))]+[0 for j in range(len(data3))])
	clf3= SVC()
	clf3.fit(data2+data3,[-1 for j in range(len(data2))]+[0 for j in range(len(data3))])
	pred = [0 for in range(len(traindata))]
	correct=0
	for i in range(len(testdata)):
		prediction=clf1.predict(testdata[i])
		if(prediction==1):
			prediction2=clf2.predict(testdata[i])
		else:
			prediction2=clf3.predict(testdata[i])
		if(prediction2==label[i]):
			correct+=1
	print "Accuracy: %.5f"%(correct/len(testdata))
	print "One versers Rest"
	clf4= SVC()
	clf4.fit(data1+data2+data3,[1 for j in range(len(data1))]+[0 for j in range(len(data2)+len(data3))])
	clf5= SVC()
	clf5.fit(data2+data1+data3,[1 for j in range(len(data2))]+[0 for j in range(len(data3)+len(data1))])
	clf6= SVC()
	clf6.fit(data3+data1+data2,[1 for j in range(len(data3))]+[0 for j in range(len(data1)+len(data2))])
	pred = [0 for in range(len(traindata))]
	correct=0
	for i in range(len(testdata)):
		prediction1=clf4.predict(testdata[i])
		prediction2=clf5.predict(testdata[i])
		prediction3=clf6.predict(testdata[i])
		if(prediction1==1 and prediction2==0 and prediction3==0):
			final=1
		elif(prediction1==0 and prediction2==1 and prediction3==0):
			final=-1
		elif(prediction1==0 and prediction2==0 and prediction3==1):
			final=0
		elif(prediction1==1 and prediction2==1):
			final=clf1.predict(testdata[i])
		elif(prediction2==1 and prediction3==1):
			final=clf2.predict(testdata[i])
		else:
			final=clf3.predict(testdata[i])
		if(final==label[i]):
			correct+=1
	print "Accuracy: %.5f"%(correct/len(testdata))
