import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#from nltk.text import TextCollection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import binary_crossentropy
#from tensorflow_addons.metrics import F1Score
#from tensorflow.keras.metrics import F1Score



data=pd.read_csv('onion-or-not.csv')
y=data['label']
#print(y)
data['tokens']=data.apply(lambda x: word_tokenize(x['text']),axis=1)

stemmer=PorterStemmer()
data['stemmed']=data['tokens'].apply(lambda x: [stemmer.stem(y) for y in x])

stop=set(stopwords.words('english'))
data['stop']=data['stemmed'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))

tf=TfidfVectorizer()
v=tf.fit_transform(data['stop'].to_numpy())
feature_names = tf.get_feature_names()
dense=v.todense()
#df = pd.DataFrame(dense, columns=[feature_names])
df = pd.DataFrame(dense)
print(df)

#train_X, val_X, train_y, val_y = train_test_split(df, y, train_size=0.75, test_size=0.25, random_state=0)
model=Sequential()
model.add(Dense(100, input_dim=16998, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=binary_crossentropy,
              optimizer='adam',
              metrics=['accuracy','Precision','Recall'])#todo f1score

model.fit(df, y,epochs=5, validation_split=0.25)
#prediction = model.predict_classes(val_X)
#prediction = np.argmax(model.predict(val_X), axis=-1)
#print(precision_recall_fscore_support(val_y,prediction))

'''
clf = MLPClassifier(random_state=1, max_iter=300).fit(train_X, train_y)
predictions=clf.predict(val_X)
print(precision_recall_fscore_support(val_y,predictions))
'''


# https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments

