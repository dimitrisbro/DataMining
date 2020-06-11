import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.cluster import KMeans


def modelSVM(train_X, val_X, train_y, val_y):
    model1 = svm.SVC(kernel='linear')
    model1.fit(train_X, train_y)
    prediction1 = model1.predict(val_X)
    p=precision_recall_fscore_support(prediction1, val_y, average='micro')
    print("accuracy:\t",accuracy_score(prediction1, val_y),"\tprecision:\t ",p[0],"\trecall:\t ",p[1],"\tfscore:\t ",p[2])
    #print("accuracy ",accuracy_score(prediction1, val_y))


def dropCol():
    # watch: Drop column
    dataDrop = nData
    dataDrop = dataDrop.drop(['pH'], axis=1)
    y = dataDrop['quality']
    X = dataDrop.drop(['quality'], axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
    print("Dropping Column")
    modelSVM(train_X, val_X, train_y, val_y)


def meanVal():
    # watch: Filling with Mean Value
    meanData = nData
    #print(meanData['quality'].unique())
    a = meanData.groupby('quality')['pH'].mean()
    for i in index:
        meanData.loc[i, 'pH'] = a[meanData.loc[i]['quality']]
    # print(meanData.isnull().sum().sum())#==0
    y = meanData['quality']
    X = meanData.drop(['quality'], axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
    print("Filling with Mean Value")
    modelSVM(train_X, val_X, train_y, val_y)


def logReg():
    # watch: Filling with Logistic Regression
    logR = LogisticRegression(max_iter=10000, random_state=0)  # Convergence Warning
    train_X = nData.drop(index, axis=0)
    val_X = nData.loc[index.sort_values()].drop(['pH'], axis=1)
    train_y = train_X['pH']
    train_X = train_X.drop(['pH'], axis=1)

    lab_enc = LabelEncoder()
    lab = lab_enc.fit_transform(train_y)
    logR.fit(train_X, lab)
    pred = lab_enc.inverse_transform(logR.predict(val_X))
    train_X['pH'] = train_y
    val_X['pH'] = pred
    logData = pd.concat([train_X, val_X]).sort_index()
    # print(logData)
    y = logData['quality']
    X = logData.drop(['quality'], axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
    print("Filling with Logistic Regression")
    modelSVM(train_X, val_X, train_y, val_y)


def kMeans():
    # watch: Filling with k-Means
    numOfClusters = len(nData['quality'].unique())
    train_X = nData.drop(index, axis=0)
    val_X = nData.loc[index.sort_values()].drop(['pH'], axis=1)
    # val_X=val_X.drop(['quality'],axis=1)
    train_y = train_X['pH']
    train_X = train_X.drop(['pH'], axis=1)
    kMeansModel = KMeans(n_clusters=numOfClusters, random_state=0).fit(train_X, train_y)
    pred = kMeansModel.predict(val_X)
    pHkValues = kMeansModel.cluster_centers_[:, 8]
    val_X['pH'] = pHkValues[pred]
    train_X['pH'] = train_y
    kData = pd.concat([train_X, val_X]).sort_index()
    y = kData['quality']
    X = kData.drop(['quality'], axis=1)
    train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
    print("Filling with k-Means")
    modelSVM(train_X, val_X, train_y, val_y)


rawData = pd.read_csv('winequality-red.csv')
y = rawData['quality']
X = rawData.drop(['quality'], axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
print("Without Missing Values")
modelSVM(train_X, val_X, train_y, val_y)
#print(rawData)
# watch: Sampling
nData = rawData
index = nData.sample(frac=0.33, random_state=1).index
nData.loc[index, 'pH'] = np.nan
# print(nData['pH'].loc[index])
# print(index)
dropCol()
meanVal()
logReg()
kMeans()
