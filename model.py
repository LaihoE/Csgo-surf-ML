import pandas as pd

bigboi=pd.read_csv(r"C:\Users\emill\PycharmProjects\OWcheaters\goodruns/cords30.csv")
for i in range(30,110):
    df2=pd.read_csv(r"C:\Users\emill\PycharmProjects\OWcheaters\goodruns/cords31.csv")
    bigboi = pd.concat([bigboi, df2], ignore_index=True)

print(bigboi)


import pandas as pd
from os import path
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from catboost import CatBoostClassifier
import keys
df=bigboi

newkeys=[]
for x in df['keys']:
    newkeys.append(x[2:3])

df["newkeys"]=newkeys
#df=df.append(newkeys)
df=df.drop("keys",axis=1)
print(df)

df["newkeys"]=df["newkeys"].replace("A",0)
df["newkeys"]=df["newkeys"].replace("W",1)
df["newkeys"]=df["newkeys"].replace("D",2)
df["newkeys"]=df["newkeys"].replace("F",3)

print(df)



X=df.drop('newkeys',axis=1)
y=df['newkeys']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#cat_features=["newkeys"]


cb_model = CatBoostClassifier(iterations=500,
                             depth=6,
                             task_type="CPU",
                             eval_metric='Accuracy',
                             random_seed=42,
                             bagging_temperature=0.2,
                             od_type='Iter',
                             metric_period=50,
                             od_wait=10,
                             learning_rate=0.0555
                             )
cb_model.fit(X_train, y_train, verbose=0)

"""random_search = RandomizedSearchCV(cb_model, param_distributions=params,verbose=5)
random_search.fit(X_train,y_train,verbose=0)
print(random_search.best_params_)
print(random_search.best_score_)"""

# r2
y_pred = cb_model.predict(X_test)
#r2test = r2_score(y_test, y_pred)
#print("CatBoost r2:", r2test)

prd=cb_model.predict([-112.442970,6292.729492,9989.057617,94.806831,30.937723])
print(prd)

import time
import pyautogui
from tset import key_check
import csv
import os
def getpos(filenumber):
    pyautogui.scroll(10)
    with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log","r") as f:
        lines=f.read().splitlines()
        last_line = lines[-1]
        splitted=last_line.split(";")
        try:
            coords=splitted[0]
            coords=coords.split(" ")
            angles = splitted[1]
            angles = angles.split(" ")
            class coordinates:
                X=coords[1]
                Z=coords[2]
                Y=coords[3]
                vertical=angles[1]
                horizontal=angles[2]
                keyspressed=key_check()
            predicted=cb_model.predict([float(coordinates.X), float(coordinates.Y), float(coordinates.Z),float(coordinates.horizontal), float(coordinates.vertical)])

            print(coordinates.X, coordinates.Y, coordinates.Z,coordinates.horizontal, coordinates.vertical,coordinates.keyspressed,"PREDICTED:",predicted)
            keys.Keys.directKey("a")
            time.sleep(0.04)
            keys.Keys.directKey("a", keys.key_release)
        except:
            pass



time.sleep(3)
for cnt in range(100):
    # Get amount of files
    path, dirs, files = next(os.walk(r"C:\Users\emill\PycharmProjects\OWcheaters/Cheaters"))
    file_count = len(files)
    import pandas as pd
    # Clear the log file to avoid it becoming too large (maybe slower to read if file is bigger?)
    with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log", "w") as f:
        f.close()
    now=time.time()
    print(now)

    x="go"
    while x != "stop":
        x=getpos(file_count)
    end=time.time()
    duration=end-now
    print(duration)
    df=pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/Cheaters/cords{file_count}.csv")
    print(len(df))
    print(len(df)/duration)
    time.sleep(1)