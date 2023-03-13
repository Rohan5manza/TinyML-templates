#Linear regression and SVM

#load downloaded dataset, drop some useless features like furnace,etc. Keep it simple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('Energy monitoring.csv')

X-train,X_test,y_train,y_test=train_test_split(data[])