import pandas as pd
import numpy as np
import matplotlib as plt
import pickle

#Read the data from file.
alz = pd.read_csv("Data Sets\\oasis_longitudinal.csv")
alz2 = pd.read_csv("Data Sets\\oasis_cross-sectional.csv")

#fill the empty spaces using the method 'ffill'
alz = alz.fillna(method='ffill')
alz2 = alz2.fillna(method='ffill')

#Drop unwanted features
alz.drop(['MRI ID'], axis=1, inplace=True)
alz.drop(['Visit'], axis=1, inplace=True)

#Replace Numerical values to Alphabetical representation
alz['CDR'].replace(to_replace=0.0, value='A', inplace=True)
alz['CDR'].replace(to_replace=0.5, value='B', inplace=True)
alz['CDR'].replace(to_replace=1.0, value='C', inplace=True)
alz['CDR'].replace(to_replace=2.0, value='D', inplace=True)

#Encode labels with values between 0 - 1 using LabelEncoder
#Split data as Train and Test using train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Encode labels for Dataset 1
for x in alz.columns:
    f = LabelEncoder()
    alz[x] = f.fit_transform(alz[x])

#Encode labels for Dataset 2
for x in alz2.columns:
    f = LabelEncoder()
    alz2[x] = f.fit_transform(alz2[x])

#combine Dataset 1 and 2
df3 = pd.concat([alz,alz2],sort = False)
df3 = df3.fillna(method = 'ffill')

#Split the data as Train and Test
train,test = train_test_split(alz,test_size=0.25)

#Split the features used for Train
X_train = train[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_train = train.CDR

#Split the features used for Test
X_test = test[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_test = test.CDR

#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

y_train=np.ravel(y_train)
X_train=np.asarray(X_train)
y_test=np.ravel(y_test)
X_test=np.asarray(X_test)

#Import KNeighborsClassifier and then classify the data
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(X_train, y_train)

#predict
prediction = knn.predict(X_test)

#Store the classifed data.
filename = 'kNeighbours.pickle'
pickle.dump(knn, open(filename, 'wb'))
