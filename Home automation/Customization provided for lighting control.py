# included both linear regresssion, and neural networks, to give user choice between different models for customization

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('Energy monitoring.csv')

#Allow users to input hyperparameters and features
#can add more hyperparameters later as command line arguments

num_features=int(input("Enter the number of features: "))
test_size=float(input('Enter the test size:'))
random_state=int(input('Enter the random state: '))

X_train,X_test,y_train,y_test=train_test_split(data.iloc[:,1:num_features+1],data['energy_consumption'],test_size=test_size,random_state=random_state)

#Allow user to input more features

while True:
    add_feature=input('Do you want to add more features (y/n): ')
    if (add_feature.lower()=='n'):
        break
    elif (add_feature.lower()=='y'):
        feature=input('Enter the name of the feature: ')
        X_train[feature]=data[feature][:len(X_train)]
        X_test[feature]=data[feature][:len(X_test)].reset_index(drop=True)

        #reset._index() is used to reset index of DataFrame after dropping a target column. After dropping 
        #unused column, index values are not updated, hence it is used to drop existing column and replace it 
        #with a new one.drop=true prevents the old index from being added as a new column in the dataframe

#following two lines are for when user wants regression instead of neural network
model=LinearRegression()
model.fit(X_train,y_train)

# below code is for when user wants to use neural network instead of regression

while True:
    add_layer=input('Do you want to add more layers to the neural network?(y/n): ')
    if add_layer.lower()=='n':
        break
    elif add_layer.lower=='y':
        n_neurons=int(input("Enter number of neurons you want to add: "))
        model.add(Dense(n_neurons,activation='relu'))

y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
print('Mean Squared Error:', mse)







