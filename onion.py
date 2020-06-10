import pandas as pd
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#from nltk.text import TextCollection
from sklearn.feature_extraction.text import TfidfVectorizer


data=pd.read_csv('onion-or-not.csv')
data['tokens']=data.apply(lambda x: word_tokenize(x['text']),axis=1)

stemmer=PorterStemmer()
data['stemmed']=data['tokens'].apply(lambda x: [stemmer.stem(y) for y in x])

stop=set(stopwords.words('english'))
data['stop']=data['stemmed'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))


tf=TfidfVectorizer()
#v=tf.fit_transform([data['stop'].loc[1],data['stop'].loc[2]])
#feature_names = tf.get_feature_names()
#dense=v.todense()
#dl=dense.tolist()
#df = pd.DataFrame(dense, columns=[data['stop'].loc[1]])
x=tf.fit_transform(data['stop'])
print(tf.vocabulary)
# https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments

