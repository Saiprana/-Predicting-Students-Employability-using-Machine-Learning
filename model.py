import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from google.colab import files
uploaded = files.upload()
import io
df2 = pd.read_csv(io.BytesIO(uploaded['database1.csv']))
df 2
df2.loc[df2["Degree"] == "Btech", "Degree"] = 0
df2.loc[df2["Degree"] == "B.E", "Degree"] = 1
df2.loc[df2["Degree"] == "Mtech", "Degree"] = 2
df 2
df2.loc[df2["Output"] == "NO", "Output"] = 0
df2.loc[df2["Output"] == "YES", "Output"] = 1
df2
df2.isnull().sum()
df2.loc[df2["Gender"] == "M", "Gender"] = 0
df2.loc[df2["Gender"] == "F", "Gender"] = 1
df2.loc[df2["Gender"] == "NaN", "Gender"] = 2
df2
df2=df2.drop(["Name","Date of Birth"],axis=1)
df2
df2["Gender"] = df2["Gender"].fillna(df2["Gender"].median())
df2
df2=df2.drop(["College Name"],axis=1)
df2
df2=df2.drop(["Email ID"],axis=1)
df2=df2.drop(["Extra curricular activities"],axis=1)
df2
df2.isnull().sum()
df2['SSC']=df2['SSC'].fillna(df2['SSC'].median())
df2['Twelth']=df2['Twelth'].fillna(df2['Twelth'].median())
df2['CGPA']=df2['CGPA'].fillna(df2['CGPA'].median())
df2['Clubs']=df2['Clubs'].fillna(df2['Clubs'].median())
df2['Output']=df2['Output'].fillna(df2['Output'].median())
df2
X = df2.drop('Output', axis=1)
y = df2['Output']
df2
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size = 0.2,\
random_state=156)
from sklearn.linear_model import LogisticRegression
my_model = LogisticRegression()
result = my_model.fit(X_train, y_train)
predictions = result.predict(X_test)
predictions
predictions1 = result.predict([[1,2,0,2,2,1,98,93,9.5,2]])
predictions1
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
print(confusion_mat)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
confusion_df=pd.DataFrame(confusion_mat,index=['Actual neg','Actual pos'],column
s=['predicted neg','predicted pos'])
confusion_df
from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))
predictionsNew1 = result.predict([[1,2,0,2,2,1,78,53,5.9,0]])
predictionsNew1
from sklearn.neighbors import KNeighborsClassifier
my_model=KNeighborsClassifier(n_neighbors=1)
result=my_model.fit(X_train,y_train)
predictions=result.predict(X_test)
predictions
predictions2 = result.predict([[1,2,0,2,2,1,91,96,8.5,2]])
predictions2
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
print(confusion_mat)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
confusion_df=pd.DataFrame(confusion_mat,index=['Actual neg','Actual pos'],column
s=['predicted neg','predicted pos'])
confusion_df
from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))
predictionsNew2 = result.predict([[1,2,0,2,2,1,74,56,5.5,1]])
predictionsNew2
from sklearn.ensemble import RandomForestClassifier
my_model=RandomForestClassifier(n_estimators=42,criterion='entropy',random_stat
e=0)
result=my_model.fit(X_train,y_train)
predictions=result.predict(X_test)
predictions
predictions3 = result.predict([[1,2,0,2,2,1,91,96,9.5,2]])
predictions3
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
print(confusion_mat)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
confusion_df=pd.DataFrame(confusion_mat,index=['Actual neg','Actual pos'],column
s=['predicted neg','predicted pos'])
confusion_df
from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))
predictionsNew3 = result.predict([[1,2,0,2,2,1,76,56,5.6,1]])
predictionsNew3
from sklearn.tree import DecisionTreeClassifier
my_model=DecisionTreeClassifier(random_state=0)
result=my_model.fit(X_train,y_train)
predictions=result.predict(X_test)
predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
print(confusion_mat)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions)
confusion_df=pd.DataFrame(confusion_mat,index=['Actual neg','Actual pos'],column
s=['predicted neg','predicted pos'])
confusion_df
from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions))
from sklearn.svm import SVC
my_model4= SVC(kernel = 'rbf', random_state = 0)
result1 = my_model4.fit(X_train, y_train)
result1
predictions2 = result1.predict(X_test)
predictions2
predictions5 = result.predict([[1,2,0,2,2,1,98,99,9.5,6]])
predictions5
|from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions2)
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,predictions2)
confusion_df=pd.DataFrame(confusion_mat,index=['Actual neg','Actual pos'],column
s=['predicted neg','predicted pos'])
confusion_df
from sklearn import metrics
print('\n**Classification Report:\n',metrics.classification_report(y_test,predictions2))