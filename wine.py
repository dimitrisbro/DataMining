import pandas as pd

rawData = pd.read_csv('/home/db/Documents/code/dataMining/winequality-red.csv')
print(rawData.describe())