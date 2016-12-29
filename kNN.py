import csv 
import random
import math
import operator 
import random
import numpy as np 
from sklearn.utils import shuffle
def loadDataSet(filename,split,trainingSet=[],testSet=[]):
	with open(filename,'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataSet=list(lines)
		np.random.shuffle(dataSet)
		for x in range(len(dataSet)-1):
			for y in range(16):
				dataSet[x][y] = float(dataSet[x][y])
			if random.random() < split:
				trainingSet.append(dataSet[x])
			else:
				testSet.append(dataSet[x])

##load the data first, then divide into training and test set 


def calcDistance(point1,point2,length):
	distance=0
	for x in range (length):
		distance +=pow((point1[x]-point2[x]),2)
	return math.sqrt(distance)
## calcDistance function used to calculate the euclidean distance between two points 



def getNeighbours(trainingSet,testInstance,k):
	distances=[]
	length=len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = calcDistance(testInstance,trainingSet[x],length)
		distances.append((trainingSet[x],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours=[]
	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours


def getResponse(neighbours):
	classVotes={}
	for x in range(len(neighbours)):
		response=neighbours[x][0]
		if response in classVotes:
			classVotes[response]+=1
		else:
			classVotes[response]=1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def main():
	trainingSet=[]
	testSet=[]
	split=0.60
	correct =0
	loadDataSet('leaf.csv', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbours(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][0]))
		
		if testSet[x][0] == result:
			correct += 1

	print('Correct = ' + repr(correct))
	print('Accuracy= ' +repr(float(correct/float(len(testSet))*100.00)) + '%') 
main()