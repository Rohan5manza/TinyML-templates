import pandas as pandas
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#loading the WISDM dataset
#save the dataset as csv file named WISDM.txt, in the same directory as this script

data=pd.read_csv('WISDM.txt',header=None,names=['user_id','activity','timestamp','x','y','z'])

data=data.dropna()
data['activity']=data['activity'].map({'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5})

X= data[['x','y','z']]

y=data['activity']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

clf=SVC(kernel='rbf') #try out other kernels later to see performance

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")