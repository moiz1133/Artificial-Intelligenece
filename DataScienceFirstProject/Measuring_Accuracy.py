import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('music1.csv')
X=df.drop(columns=['genre'])
Y=df['genre']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(X_train, y_train)
prediction=model.predict(X_test)
score=accuracy_score(y_test,prediction)
score