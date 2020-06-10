import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,scale
from sklearn.cluster import KMeans

rawData = pd.read_csv('winequality-red.csv')
#print(rawData)
#watch: Sampling
nData=rawData
index = nData.sample(frac=0.33, random_state=1).index
nData.loc[index, 'pH'] = np.nan
#print(nData['pH'].loc[index])
#print(index)
'''
#watch: Drop column
dataDrop=nData
dataDrop=dataDrop.drop(['pH'],axis=1)

#watch: Replace with mean
meanData=nData
print(meanData['quality'].unique())
a=meanData.groupby('quality')['pH'].mean()
for i in index:
    meanData.loc[i,'pH']=a[meanData.loc[i]['quality']]
#print(meanData.isnull().sum().sum())#==0

'''
'''
#watch: Fill with Logistic Regression
logR=LogisticRegression()
train_X=nData.drop(index,axis=0)
val_X=nData.loc[index.sort_values()].drop(['pH'],axis=1)
train_y=train_X['pH']
train_X=train_X.drop(['pH'],axis=1)

lab_enc=LabelEncoder()
lab=lab_enc.fit_transform(train_y)
logR.fit(train_X,lab)
#pred=lab_enc.inverse_transform(logR.predict(val_X))
#print(pred)


'''
#watch: Kmeans
kmeans = KMeans(n_clusters=10, random_state=0).fit(train,y)
pred=kmeans.predict(val)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
