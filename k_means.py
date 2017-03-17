import cv2
import numpy as np
import os
import glob
import csv
import argparse
import math
from scipy.cluster.vq import *

k=3
l=6
no_of_centroids=int((math.pow(k,l-1)-1)/(k-1))
centroids=np.zeros((no_of_centroids,k,128),"float32")
def go(deslist,n):
	BOW = cv2.BOWKMeansTrainer(k)

	if(n>=no_of_centroids):
		return
	if len(deslist)<k:
		return	
	centroids[n]=BOW.cluster(deslist)
	print(centroids[n])
	'''
	for i in range(len(deslist)):
		word, dis=vq(np.array(deslist[i]),centroids[n])
		if word==0:
			des0.append(deslist[i])
		if word==1:
			des1.append(deslist[i])
		if word==2:
			des2.append(deslist[i])

	'''
	words,dis=vq(deslist,centroids[n])
	des0=[]
	des1=[]
	des2=[]
	des3=[]
	c=0
	for w in words:
		if(w==0):
			des0.append(deslist[c])
		if(w==1):
			des1.append(deslist[c])
		if(w==2):
			des2.append(deslist[c])
		
		if(w==3):
			des3.append(deslist[c])	
		
		c=c+1




	go(np.array(des0),n*k+1)
	go(np.array(des1),n*k+2)
	go(np.array(des2),n*k+3)
	go(np.array(des3),n*k+4)





def go1(d,n):
	if n>=(math.pow(k,l-1)-1)/(k-1):
		return n-int((math.pow(k,l-1)-1)/(k-1))

	
	temp=[]
	temp.append(d)
	temp=np.array(temp)
	word,dis=vq(temp,centroids[n])

	return go1(d,n*k+word[0]+1)





ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())







#dictionarySize = 100
#BOW = cv2.BOWKMeansTrainer(dictionarySize)


des_list=[]
cnt=0
imageID=[]
directory=os.listdir(args["dataset"])
print(directory)

labels=[]
label_count=0
im_count=0
x=[]
for d in directory:

	for p in glob.glob(args["dataset"]+'/'+d + "/*.jpg"):
		labels.append(label_count)
		im_count=im_count+1

		imageID.append(p);

		print(p)
		#print(p.split("/")[2])
		cnt=cnt+1
		image = cv2.imread(p)

		'''
		height,width,depth = image.shape
		circle_img = np.zeros((height,width), np.uint8)
		strn=p.split("/")[2]
		#x.split("_")[0]
	
		#print(x.split("_")[0])
		#x=input()
		if(strn.split("_")[0]=="N3"):
			#print(p)
			#x=input()
			cv2.circle(circle_img,(int(width/2)-10,int(height/2)-50),100,1,thickness=-1)
			masked_data = cv2.bitwise_and(image, image, mask=circle_img)	
		elif(strn.split("_")[0]=="N2"):
			#print(p)
			#x=input()
			cv2.circle(circle_img,(int(width/2),int(height/2)-50),100,1,thickness=-1)
			masked_data = cv2.bitwise_and(image, image, mask=circle_img)	
		else:	
			cv2.circle(circle_img,(int(width/2),int(height/2)-7),100,1,thickness=-1)
			masked_data = cv2.bitwise_and(image, image, mask=circle_img)
		'''
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp, dsc= sift.detectAndCompute(gray, None)
		y=[]
		#x=[]
		for de in dsc:
			x.append(de)
			y.append(de)
		des_list.append(y)	
		#des_list.append(x)
		#BOW.add(dsc)
	label_count=label_count+1

'''
for p in glob.glob(args["dataset"]+ "/*.jpg"):
	imageID.append(p);
	print(p)
	cnt=cnt+1
	image = cv2.imread(p)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, dsc= sift.detectAndCompute(gray, None)
	x=[]
	for de in dsc:
		x.append(de)
	des_list.append(x)
	BOW.add(dsc)
'''
#dictionary created


print("hola")
go(np.array(x),0)
#dictionary = BOW.cluster(x)
print("hola")





#print(dictionary)
'''
print(dictionary.shape)

output = open("dictionary.csv", "w")
for i in range(dictionarySize):
	for j in range(128):
		output.write(str(dictionary[i][j]))
		output.write(',')
	output.write('\n')
output.close()

print(type(dsc))
print(dsc.shape)
'''

output = open("centroids.csv", "w")
for it1 in range(no_of_centroids):
	for it2 in range(k):
		for it3 in range(128):
			output.write(str(centroids[it1][it2][it3]))
			output.write(',')
		output.write('\n')
output.close()


im= np.zeros((len(des_list),int( math.pow(k,l-1))), "float32")

for i in range(len(des_list)):
	
	for j in range(len(des_list[i])):
		c=go1(np.array(des_list[i][j]),0)
		im[i][c]+=1


print("Printing im")
print(im)
#x=input()
'''
im_features = np.zeros((len(des_list), dictionarySize), "float32")
words_per_image=np.zeros(len(7t))
for i in range(len(des_list)):
	words, distance = vq(des_list[i],dictionary)
	words_per_image[i]=len(words)
	#print(words)
	#print(distance)
	#x=input()
	for w in words:
		im_features[i][w] += 1

	print(im_features[i])
	print(len(words))
	#x=input()	
print(im_features)	
'''
nbr=np.zeros(int( math.pow(k,l-1)))
for i in range(len(des_list)):
	
	for j in range(int( math.pow(k,l-1))):
		if(im[i][j]!=0):
			nbr[j]=nbr[j]+1

print(nbr)
idf=np.zeros(int( math.pow(k,l-1)))
for j in range(int( math.pow(k,l-1))):
	if(nbr[j]!=0):	
		idf[j]=np.log(len(des_list)/nbr[j])
	else:
		idf[j]=np.log(len(des_list))



print(idf)
#x=input()


output = open(args["index"], "w")
output.write("idf")
output.write(',')
for w in range(int( math.pow(k,l-1))):
	output.write(str(idf[w]))
	output.write(',')
output.write('\n')

for i in range(len(des_list)):
	if(imageID[i].split("/")[2]=="N1_0.jpg" or imageID[i].split("/")[2]=="N1_33.jpg" or imageID[i].split("/")[2]=="N1_342.jpg" or imageID[i].split("/")[2]=="N2_0.jpg" or imageID[i].split("/")[2]=="N2_33.jpg" or imageID[i].split("/")[2]=="N2_342.jpg" or imageID[i].split("/")[2]=="N3_0.jpg" or imageID[i].split("/")[2]=="N3_33.jpg" or imageID[i].split("/")[2]=="N3_342.jpg"):
		continue
	
	output.write(str(imageID[i]))

	output.write(',')
	for w in range(int( math.pow(k,l-1))):
		if(im[i][w]!=0):
			im[i][w]=im[i][w]*idf[w]
		output.write(str(im[i][w]))
		output.write(',')

	output.write('\n')

output.close()

print(im)







#
#[  56.   71.   68.   37.   52.  169.   46.]