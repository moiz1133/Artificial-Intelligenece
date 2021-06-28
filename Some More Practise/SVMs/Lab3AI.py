

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
#loading the dataset
df=pd.read_csv(r'trainingSet.csv')

X_train=df.iloc[:,:-1].values
Y_train=df.iloc[:,-1].values


# Now the value is splitted into independent and dependend variables.Now training the svm model

from sklearn.svm import SVC
svm_model_linear=SVC(kernel='linear',C=1).fit(X_train,Y_train)


# Now the model is train.We are taking the X_test from the test.csv 

df=pd.read_csv(r'testSet1.csv')
X_test=df.iloc[:,:-1].values


# We are going to get our predictions

sp=svm_model_linear.predict(X_test)

arr = np.array(sp)

output = pd.DataFrame(data=arr.flatten())


writer = ExcelWriter('results.xlsx')
output.to_excel(writer,index=False)
writer.save()
