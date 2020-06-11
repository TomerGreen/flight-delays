from model import *

print('hi')
#%%
def test1():
    print("test1")

if __name__=="__main__":
    test1()

#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('train_data_10k_class.csv')

df2 = pd.DataFrame()

df2['DayOfWeek'] = df['DayOfWeek']
df2['DelayFactor'] = df['DelayFactor']

factor = pd.factorize(df2['DelayFactor'])

X = df2.drop(['DelayFactor'],axis=1)
y = df['DelayFactor']

model = RandomForestClassifier(n_estimators=10,max_depth=5)
model.fit(X,y)

# Predicting the Test set results
y_pred = model.predict(X)
#Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
y_test = y

# Making the Confusion Matrix
print(pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted']))

df3 = df.dropna()

df3.to_csv("testDf")

df4 = pd.read_csv('book1.csv')