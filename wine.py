import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

rawData = pd.read_csv('winequality-red.csv')

y = rawData['quality']
X = rawData.drop(['quality'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)
'''
model=svm.SVC(kernel='linear')
model.fit(train_X, train_y)
pred=model.predict(val_X)
#mse= -1* cross_val_score(model, val_X, val_y, cv=2, scoring='neg_mean_absolute_error')#todo cross val
#print(mse)
print(precision_recall_fscore_support(pred, val_y, average='weighted'))
#print(mean_squared_error(pred, val_y))


'''


model1 = svm.SVC(kernel='linear')
model1.fit(train_X, train_y)
prediction1 = model1.predict(val_X)
print('SVC + linear', precision_recall_fscore_support(prediction1, val_y, average='micro'))
#print(mean_squared_error(prediction1, val_y))

model2 = svm.LinearSVC(max_iter=10000)
model2.fit(train_X, train_y)
prediction2 = model2.predict(val_X)
print('LinearSVC', precision_recall_fscore_support(prediction2, val_y, average='micro'))
#print(mean_squared_error(prediction2, val_y))

model3 = svm.SVC(kernel='rbf', gamma=0.8)
model3.fit(train_X, train_y)
prediction3 = model3.predict(val_X)
print('SVC + rbf', precision_recall_fscore_support(prediction3, val_y, average='micro'))
#print(mean_squared_error(prediction3, val_y))
# '''
