# Decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


import pandas as pd

data=pd.read_csv('') #demo dataset will be loaded here

X_train,X_test,y_train,y_test = train_test_split(data[['occupancy','time_of_day']],data['light_on'],test_size=0.2)

clf= DecisionTreeClassifier()
clf.fit(X_train,y_train)

predictions=clf.predict(x_test)

accuracy=clf.score(X_test,y_test)

print('Accuracy: {:.2f}%'.format(accuracy*100))

#following code lines are for saving model by converting to .tflite micro

converter=tf.lite.TFLiteConverter.from_sklearn(clf)
tflite_model=converter.convert() # makes a byte string representation of model

#saving the file

with open('model.tflite','wb') as f:
    f.write(tflite_model)

# After storing the model in a .tflite format, a C++ program is needed to load and run it on microcontroller, using github files i downloaded

#model should run inference on new data collected from tensors

#input tensors should be in same format as training data used .

#for our example,input features are occupancy, and time_of_day,
#  hence input tensors should be a 2D array with shape(num_samples,2), where 2 is number of features


 #this converts the DataFrame into a numpy array, can directly be fed into the Invoke() method of interpreter


