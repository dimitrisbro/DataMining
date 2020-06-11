import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error



rawData = pd.read_csv('winequality-red.csv')

y = rawData['quality']
X = rawData.drop(['quality'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

param_grid = {
    # 'kernel':['linear'],
    'C':[1,5]
}
print('dogsd')

grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid = param_grid, n_jobs = -1)
grid.fit(train_X, train_y)
df=pd.DataFrame(grid.cv_results_)
print(df.columns)
#print(df[['mean_test_score','param_coef0','param_kernel','rank_test_score']])
print(df[['param_C','mean_test_score','rank_test_score']])
#print(df[['mean_test_score','param_kernel','rank_test_score']])

# models=[svm.SVC(kernel='linear',random_state=0),svm.SVC(kernel='rbf', gamma=0.7, C=10,random_state=0),svm.SVC(kernel='poly',coef0=7,random_state=0)]
# result=[]
# result1=[]
#
# for i in models:
#     i.fit(train_X, train_y)
#     result.append(precision_recall_fscore_support(i.predict(val_X), val_y, average='micro'))
#     result1.append(mean_squared_error(i.predict(val_X), val_y))
#
# for i in result:
#     print(i)
# for i in result1:
#     print(i)
# #'''