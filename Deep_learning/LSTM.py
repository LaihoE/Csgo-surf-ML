import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D


bigboi = pd.read_csv("C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords1.csv")
for i in range(50, 226):
    df2=pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords{i}.csv")
    bigboi = pd.concat([bigboi, df2], ignore_index=True)

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pickle

df=bigboi
listaX=[]
listaY=[]
listaZ=[]


for point in range(round(len(df)/100)):
    index=point*100
    batch=np.array(df.iloc[index])




listaX.append(0)
listaY.append(0)
listaZ.append(0)
for i in range(1,len(df["X"])):

    last=df["X"].iloc[i-1]
    now=df["X"].iloc[i]
    speed=now-last
    listaX.append(speed)

for i in range(1,len(df["Y"])):
    last=df["Y"].iloc[i-1]
    now=df["Y"].iloc[i]
    speed=now-last
    listaY.append(speed)

for i in range(1,len(df["Z"])):
    last=df["Z"].iloc[i-1]
    now=df["Z"].iloc[i]
    speed=now-last
    listaZ.append(speed)

df["SpeedX"]=listaX
df["SpeedY"]=listaY
df["SpeedZ"]=listaZ

print(df.columns)
df = df.drop("keys",axis=1)


dfx = df.drop("horizontal",axis=1)
dfx = dfx.drop("vertical",axis=1)



dfy = df["vertical"]




training_set = dfx.iloc[:300000].values
training_sety = dfy.iloc[:300000].values


#test_set = df.iloc[800:, 1:2].values

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, 800):
    X_train.append(training_set[i-60:i])
    y_train.append(training_sety[i])

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))
print(X_train.shape)
print(X_train[0])

model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 6)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 2)