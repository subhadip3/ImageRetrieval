# import the necessary packages
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import glob
import math
import csv
import numpy as np
from scipy.cluster.vq import *
import cv2
import math
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import os
k=3
l=6
no_of_centroids=int((math.pow(k,l-1)-1)/(k-1))
centroids=np.zeros((no_of_centroids,k,128),"float32")
def comparison(query,im_features):
	query=np.array(query,"float32")
	im_features=np.array(im_features,"float32")
	
	qsum=np.sum(np.absolute(query))
	#print(qsum)
	results=np.array(len(im_features),"float32")
	results=[]

	for i in range(len(im_features)):
		query1=np.zeros(len(query),"float32")
		#qsum=np.sum(np.absolute(query))
		'''
		for j in range(len(im_features[i])):
			if(im_features[i][j]==0):
				query1[j]=0
			else:
				query1[j]=query[j]
		'''
		qsum=np.sum(np.absolute(query))		
		
		isum=np.sum(np.absolute(im_features[i]))
		if(qsum!=0):
			v=query/qsum - im_features[i]/isum
		else:
			v=query - im_features[i]/isum
			
		#print(v.shape)
		#print(query.shape)
		vsum=np.sum(np.absolute(v))
		#print(type(vsum))
		#results[i]=vsum
		results.append(vsum)
	results=np.array(results)
	sort_index=np.argsort(results)	
	reults=sorted(range(len(results)), key=lambda k: results[k])
	print(sort_index)
	print(len(sort_index))
	#x=input()
	return sort_index
			
	
def go1(d,n):
	if n>=(math.pow(k,l-1)-1)/(k-1):
		return n-int((math.pow(k,l-1)-1)/(k-1))

	
	temp=[]
	temp.append(d)
	temp=np.array(temp)
	word,dis=vq(temp,centroids[n])

	return go1(d,n*k+word[0]+1)

 
class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
 
	def search(self, queryPath, limit =100):
		# initialize our dictionary of results
		
		No_of_visual_words=int(math.pow(k,l-1))

		results = {}
		# open the index file for reading
		
		f1=open("centroids.csv")
		r1=csv.reader(f1)
		
		it1=0
		it2=0
		for row in r1:
			it3=0

			for j in range(128):
				
				
				centroids[it1][it2][it3]=float(row[j])
				it3=it3+1
				#j=j+1
			it2=(it2+1)%k
			if(it2%k==0):
				it1=it1+1

		f1.close()

		im_features=[]
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
			#word_count=np.zeros(No_of_visual_words)
			#words, distance = vq(queryFeatures,dictionary)
			#for w in words:
			#	word_count[w] += 1
			#print(word_count)
			#x=input()	


			# loop over the rows in the index
			row_count=0
			label_count=0
			image_name=[]
			labels=np.zeros(5292)
			for row in reader:

				if (row_count==0):
					idf=np.zeros(No_of_visual_words)					
					
					for j in range(No_of_visual_words+1):
						if j==0:
							continue
						idf[j-1]=float(row[j])
					#print(No_of_visual_words)	
					#for w in range(No_of_visual_words):
					#	word_count[w]=(word_count[w]/len(words))*idf[w];

					#print(word_count)
					#x=input()				
					row_count=row_count+1
					continue



				
				features=np.zeros(No_of_visual_words)				
				for j in range(No_of_visual_words+1):
					if j==0:
						continue
					
					features[j-1]=float(row[j])
				im_features.append(features)
				
				#print(row[0])
				#print(features)
				#print(word_count)

				temp=row[0].partition('/')[-1].rpartition('/')[0]
				if(row_count!=1 and temp!=prev):
					label_count=label_count+1
				prev=temp	
				labels[row_count-1]=label_count
				image_name.append(row[0])
				row_count=row_count+1

				#x=input()	
				
				#sum1=0
				#for i in range(No_of_visual_words):
				#	sum1=sum1+(features[i])*(features[i])
				#sum2=0
				#for i in range(No_of_visual_words):
				#	sum2=sum2+(word_count[i])*(word_count[i])	
				#d=np.dot(features,word_count)/(math.sqrt(sum1)*math.sqrt(sum2))	
				
				#results[row[0]] = d
 				

			im_features=np.array(im_features)
			# close the reader
			f.close()
		'''
		c=0
		for p in glob.glob("Dataset"+'/'+"cup_noodles_shrimp_picante" + "/*.jpg"):
			if(c==0):
				image = cv2.imread(p)
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				sift = cv2.xfeatures2d.SIFT_create()
				kp, dsc= sift.detectAndCompute(gray, None)
				word_count=np.zeros(No_of_visual_words)
				words, distance = vq(dsc,dictionary)
				for w in words:
					word_count[w] += 1
				print(word_count)
				print(len(words))
				print(idf)
				c=c+1	

		'''
		svc = LinearSVC()
		clf = CalibratedClassifierCV(svc, cv=10)
		clf.fit(im_features, labels)
		print(len(im_features))
		print(len(labels))

		directory=os.listdir(queryPath)
		#print(directory)
		all_query_features=[]
		folder_cnt=0
		test_labels=[]

		#for d in directory:

		for p in glob.glob(queryPath+ "/*.jpg"):
			image = cv2.imread(p)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			sift = cv2.xfeatures2d.SIFT_create()
			kp, dsc= sift.detectAndCompute(gray, None)
			print(p)
			
			word_count=np.zeros(No_of_visual_words)	
			for j in range(len(dsc)):
				c=go1(np.array(dsc[j]),0)	
				word_count[c] += 1
			
			#for j in range(No_of_visual_words):
			#	print(word_count[j])
			#print(len(dsc))
			#x=input()	

			for w in range(No_of_visual_words):
				word_count[w]=word_count[w]*idf[w];
				#print(word_count[w])

			#print(word_count)
			#x=input()		
			x=comparison(word_count,im_features)
			#print(type(x))
			name=str(p).split(".")[0]+".txt"
			name=name.split("/")[1]
			#print(name)
			#x=input()


			

			file=open(name,"w")
			#r=csv.reader(file)
			for i in range(len(x)):
				category=image_name[x[i]].split("/")[1]
				im_No=image_name[x[i]].split("/")[2]
				#print(str(image_name[x[i]]))
				file.write(im_No+" ")
				file.write(category)
				
				file.write("\n")
			file.close()	
			
			'''
			
			proba = clf.predict([word_count])
			print(proba)
			print(image_name[int(proba*63)])
			

			#all_query_features.append(word_count)
			#test_labels.append(folder_cnt)
				
			#print(p)
			#folder_cnt+=1
		#test_labels=np.array(test_labels)
		#np.array(all_query_features)
		'''
		'''
		svc = LinearSVC()
		clf = CalibratedClassifierCV(svc, cv=10)
		clf.fit(im_features, labels)
		proba = clf.predict(all_query_features)
		
		print("check")
		print(im_features[0])
		print(all_query_features[0])
		'''
		'''
		#print(image_name[int(proba*72)])
		'''
		#print(proba)
		#print(labels)
		'''
		#print(accuracy_score(test_labels, proba))
		

		
		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		#results = sorted([(v, k) for (k, v) in results.items()],reverse=True)
		
 
		# return our (limited) results
		'''
		#return results[:limit]

	
	



		