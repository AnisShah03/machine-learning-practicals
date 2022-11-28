# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:39:23 2020

@author: Vishi
"""


# Implement Decision Tree Regression  to predict  person in terms of his Age and Salary 
# whether he/she is supposed to buy the car or Not 
# according to an unknown level
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('prac8/Social_Network_Ads.csv')  
  
# Age and EstimatedSalary as our independent variable matrix.
# And take the Purchased column in the dependent variable vector. 
x= data_set.iloc[:,[2,3]].values  
y= data_set.iloc[:, 4].values 

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20, random_state=0)  

# Scaling the Datasets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Fitting the Datataset with Regression Class 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# Predicting the output 
y_pred = classifier.predict(x_test)


# Building the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score

sc1 = accuracy_score(y_test, y_pred)*100

print('The Accuracy   is  ',sc1)

#classification report-it shows 
#a representation of the main classification metrics
#on a per class basis,It gives deeper intutition 
#of the classifier,It shows the precison,recall
#F1-score and support scores for the model
from sklearn.metrics import classification_report

creport = classification_report(y_test, y_pred)
print(creport)

from sklearn.metrics import confusion_matrix  
cm1= confusion_matrix(y_test, y_pred)  
print('Cm1',cm1)
# Visualising the Training set results 
from matplotlib.colors import ListedColormap 
X_set, y_set = x_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
cmap = ListedColormap(('red', 'green'))) 
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 

for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j) 

plt.title('Logistic Regression (Training set)') 
plt.xlabel('Age') 
plt.ylabel('Estimated Salary') 
plt.legend() 
plt.show()

# Visualizing the Test set results
#Our model is well trained using the training 
#dataset. Now, we will visualize the result 
#for new observations (Test set). 
#The code for the test set will remain same
#as above except that here 
#we will use x_test and y_test 
#instead of x_train and y_train.
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
