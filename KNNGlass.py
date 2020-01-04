import pandas as pd
import numpy as nm
glassData =  pd.read_csv("C:/My Files/Excelr/12 - KNN/Assignment/glass.csv")
glassData.columns
glassData.describe()
glassData.Type.value_counts()
len(glassData.columns)

from sklearn.model_selection import train_test_split
train,test = train_test_split(glassData,test_size = 0.3)
trainX = train.iloc[:,0:8]
trainY = train.iloc[:,9]
testX = test.iloc[:,0:8]
testY = test.iloc[:,9]

from sklearn.neighbors import KNeighborsClassifier as KNN
model1 = KNN(n_neighbors=2).fit(trainX,trainY)
model1_pred = model1.predict(trainX)
accurancy_m1 = nm.mean(model1_pred == trainY)
accurancy_m1

model2 = KNN(n_neighbors=5).fit(trainX,trainY)
model2_pred = model2.predict(trainX)
accurancy_m2 = nm.mean(model2_pred == trainY)
accurancy_m2

accurancy_ar=[];

for i in range(5,50):
    modeli = KNN(n_neighbors=i).fit(trainX,trainY)
    modeli_train_pred = modeli.predict(trainX)
    modeli_test_pred = modeli.predict(testX)
    train_acc = nm.mean(modeli_train_pred == trainY) 
    test_acc = nm.mean(modeli_test_pred == testY) 
    accurancy_ar.append([train_acc,test_acc])

import matplotlib.pyplot as plt
plt.plot(nm.arange(5,50),[i[0] for i in accurancy_ar],"bo-")
plt.plot(nm.arange(5,50),[i[1] for i in accurancy_ar],"ro-")
plt.legend(["train","test"])
