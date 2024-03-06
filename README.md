# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

In this model, linear regression is employed to forecast values for unknown data. The architecture includes an input layer with 16 neurons, utilizing one column as input, and a subsequent hidden layer with 16 neurons. Both layers are equipped with rectified linear unit (ReLU) activation functions. The final output layer consists of one neuron responsible for providing the predicted value. While regression models aim to capture relationships between variables, it's acknowledged that perfect fits to the data may not always be achieved. Neural networks, despite their computational demands, offer versatility in selecting regression types. The inclusion of hidden layers in these networks can further enhance their predictive capabilities.

## Neural Network Model
![image](https://github.com/S-Priyadharshan/basic-nn-model/assets/145854138/f7610211-35a4-4be3-a6f3-a32b770ac5da)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Priyadharshan S
### Register Number: 212223240127
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('datasetdl').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float','OUTPUT':'float'})
df.head()

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


X = df[['INPUT']].values
y = df[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 20)


Scaler = MinMaxScaler()

Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)


model=tf.keras.Sequential()

model.add(Dense(16,activation='relu',input_shape=(1,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))


model.compile(optimizer='adam',loss='mae')


model.fit(X_train1,y_train,epochs=1000)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1 = [[25.0]]


X_n1_1 = Scaler.transform(X_n1)


model.predict(X_n1_1)



```
## Dataset Information

![image](https://github.com/S-Priyadharshan/basic-nn-model/assets/145854138/acd421be-39e4-49d0-ba86-91cee9e3ab21)


## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/S-Priyadharshan/basic-nn-model/assets/145854138/cb6f4f4d-b3d0-4e92-99ee-ba0261636f35)



### Test Data Root Mean Squared Error
![image](https://github.com/S-Priyadharshan/basic-nn-model/assets/145854138/4e7f7f90-7285-4da8-b3f2-2de6941e5aca)


### New Sample Data Prediction

![image](https://github.com/S-Priyadharshan/basic-nn-model/assets/145854138/69979adf-ab6e-408e-8af6-6f552ad166a7)


## RESULT
A neural network regression model for the given dataset has been developed successfully.

