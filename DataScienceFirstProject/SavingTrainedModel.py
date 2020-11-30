import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
# music_data=pd.read_csv('music1.csv')
# X=music_data.drop(columns=['genre'])
# y=music_data['genre']
# model=DecisionTreeClassifier()
# model.fit(X, y)
model=joblib.load('MyFile.joblib')
predictions=model.predict([[21,1]])
predictions