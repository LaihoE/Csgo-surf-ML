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

npx=np.array([0,0,0])


for point in range(round(len(df)/100)):
    index=point*100
    batch=np.array(df.iloc[index])
    print(batch)





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

print(df)