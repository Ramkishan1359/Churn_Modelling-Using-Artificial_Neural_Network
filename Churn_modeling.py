import numpy as np
import os
import pandas as pd 
os.getcwd()
os.chdir('C:/Users/saath/Downloads')
dataset=pd.read_csv('Churn_Modelling.csv')
dataset.head()
dataset.shape
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])
X
X[:,2]=le.fit_transform(X[:,2])
X
tempdf=pd.DataFrame(X)

oh=OneHotEncoder(categorical_features=[1])
X=oh.fit_transform(X).toarray()
X
tempdf=pd.DataFrame(X)
X=X[:,1:]
X.shape

from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
lr.fit(X_train,Y_train)
Y_pred=lr.predict(X_test)
Y_pred

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)




import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


classifier=Sequential()
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

classifier.add(Dense(units=10,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=10,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=20,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()

classifier.fit(X_train,Y_train,batch_size=32,epochs=100)
Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
cm
score=classifier.evaluate(X_test,Y_test)
score








