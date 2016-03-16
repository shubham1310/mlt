import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
path='../../data/assign3/connect-4.data'
f=open(path,'r')
f=f.read()

def adddata(stri,data1,data2,data3):
	if stri=='':
		return [data1,data2,data3]
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
		label.append(a)
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
	correct=0
	for j in range(len(testdata)):
		prediction1=clf1.predict(testdata[j])
		prediction2=clf2.predict(testdata[j])
		prediction3=clf3.predict(testdata[j])
		if(prediction1==1 and prediction2 == 1):
			final=1
		elif(prediction1==-1 and prediction3 == -1):
			final=-1
		else:
			final=0
		if(final==label[j]):
			correct+=1
	print "Accuracy: %.5f"%(correct*1.0/len(testdata))
	print "One versers Rest"
	clf4= SVC(probability=True)
	clf4.fit(data1+data2+data3,[1 for j in range(len(data1))]+[0 for j in range(len(data2)+len(data3))])
	clf5= SVC(probability=True)
	clf5.fit(data2+data1+data3,[1 for j in range(len(data2))]+[0 for j in range(len(data3)+len(data1))])
	clf6= SVC(probability=True)
	clf6.fit(data3+data1+data2,[1 for j in range(len(data3))]+[0 for j in range(len(data1)+len(data2))])
	correct=0
	for j in range(len(testdata)):
		final=-3
		prediction1=clf4.predict(testdata[j])
		prediction2=clf5.predict(testdata[j])
		prediction3=clf6.predict(testdata[j])
		if(prediction1==1 and prediction2==0 and prediction3==0):
			final=1
		elif(prediction1==0 and prediction2==1 and prediction3==0):
			final=-1
		elif(prediction1==0 and prediction2==0 and prediction3==1):
			final=0
		if(final==label[j]):
			correct+=1
		elif(final==-3):
			if(prediction1==1 and prediction2==1):
				a=clf4.predict_proba(testdata[j])[0]
				b=clf5.predict_proba(testdata[j])[0]
				if(a[1]>b[1]):
					final=1
				else:
					final=-1
			elif(prediction2==1 and prediction3==1):
				a=clf5.predict_proba(testdata[j])[0]
				b=clf6.predict_proba(testdata[j])[0]
				if(a[1]>b[1]):
					final=-1
				else:
					final=0
			else:
				a=clf4.predict_proba(testdata[j])[0]
				b=clf6.predict_proba(testdata[j])[0]
				if(a[1]>b[1]):
					final=1
				else:
					final=0
			if(final==label[j]):
				correct+=1
	print "Accuracy: %.5f"%(correct*1.0/len(testdata))
