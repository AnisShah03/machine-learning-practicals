from re import T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv("prac1\Social_Network_Ads.csv")

x= data_set.iloc[:,[2,3]].values  
y= data_set.iloc[:, 4].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)
print('len',len(y_test))


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)
print(x_test)


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)

y_pred =classifier.predict(x_test)
print('y_Pred',y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
sc1 = accuracy_score(y_test,y_pred)*100

print('The Accuracy is ',sc1)
