import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.metrics import accuracy_score
import random 
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.lda import LDA 






csvFile = open('leaf.csv','rb')
reader = csv.reader(csvFile, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x)

np.random.shuffle(data)

X_data = data[:,2:15]
Y_data = data[:,0]

no_of_points = X_data.shape[0]

trainingDataSet = 0.65*no_of_points

X_train = X_data[:trainingDataSet]
Y_train = Y_data[:trainingDataSet]

X_train = preprocessing.scale(X_train) ##normalizing data 





X_test = X_data[(trainingDataSet+1):]
Y_test = Y_data[(trainingDataSet+1):]
X_test = preprocessing.scale(X_test)
X_test = X_test.astype('float')

pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


clf = GaussianNB()
clf.fit(X_train,Y_train)

LabelPrediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test,LabelPrediction)

print accuracy * 100