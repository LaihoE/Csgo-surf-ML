import pandas as pd
bigboi = pd.read_csv(r"C:\Users\emill\PycharmProjects\OWcheaters\goodruns/cords30.csv")
for i in range(30, 110):
    df2 = pd.read_csv(r"C:\Users\emill\PycharmProjects\OWcheaters\goodruns/cords31.csv")
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

df = bigboi

newkeys = []
for x in df['keys']:
    newkeys.append(x[2:3])

df["newkeys"] = newkeys
# df=df.append(newkeys)
df = df.drop("keys", axis=1)
print(df)

df["newkeys"] = df["newkeys"].replace("A", 0)
df["newkeys"] = df["newkeys"].replace("W", 1)
df["newkeys"] = df["newkeys"].replace("D", 2)
df["newkeys"] = df["newkeys"].replace("F", 3)


df=df.drop("newkeys",axis=1)
df=df.drop("vertical",axis=1)
X = df.drop('horizontal', axis=1)
y = df['horizontal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# cat_features=["newkeys"]


cb_modelh = CatBoostRegressor(iterations=500,
                              depth=6,
                              task_type="CPU",
                              eval_metric='R2',
                              random_seed=42,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=50,
                              od_wait=10,
                              learning_rate=0.0555
                              )
cb_modelh.fit(X_train, y_train, verbose=0)

"""random_search = RandomizedSearchCV(cb_model, param_distributions=params,verbose=5)
random_search.fit(X_train,y_train,verbose=0)
print(random_search.best_params_)
print(random_search.best_score_)"""

# r2
y_pred = cb_modelh.predict(X_test)
# r2test = r2_score(y_test, y_pred)
# print("CatBoost r2:", r2test)

prd = cb_modelh.predict([-112.442970, 6292.729492, 9989.057617, 94.806831, 30.937723])
print(prd)
##############################
df = bigboi

newkeys = []
for x in df['keys']:
    newkeys.append(x[2:3])

df["newkeys"] = newkeys
# df=df.append(newkeys)
df = df.drop("keys", axis=1)
print(df)

df["newkeys"] = df["newkeys"].replace("A", 0)
df["newkeys"] = df["newkeys"].replace("W", 1)
df["newkeys"] = df["newkeys"].replace("D", 2)
df["newkeys"] = df["newkeys"].replace("F", 3)


df=df.drop("newkeys",axis=1)
df=df.drop("horizontal",axis=1)
X = df.drop('vertical', axis=1)
y = df['vertical']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# cat_features=["newkeys"]


cb_modelv = CatBoostRegressor(iterations=500,
                              depth=6,
                              task_type="CPU",
                              eval_metric='R2',
                              random_seed=42,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=50,
                              od_wait=10,
                              learning_rate=0.0555
                              )
cb_modelh.fit(X_train, y_train, verbose=0)




import time
import pyautogui
from tset import key_check
import csv
import os
from testing import Keys
keys = Keys()
def getpos(filenumber):
    keys.directKey("p")
    time.sleep(0.01)
    keys.directKey("p", keys.key_release)
    start = time.time()
    with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log",
              "r") as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        splitted = last_line.split(";")
        try:
            coords = splitted[0]
            coords = coords.split(" ")
            angles = splitted[1]
            angles = angles.split(" ")

            class coordinates:
                X = coords[1]
                Z = coords[2]
                Y = coords[3]
                vertical = angles[1]
                horizontal = angles[2]
                keyspressed = key_check()

            predicted = cb_modelh.predict(
                [float(coordinates.X), float(coordinates.Y), float(coordinates.Z), float(coordinates.horizontal)])
            difh=predicted-float(coordinates.horizontal)

            predicted = cb_modelh.predict(
                [float(coordinates.X), float(coordinates.Y), float(coordinates.Z), float(coordinates.vertical)])
            difv = predicted - float(coordinates.vertical)

            #difh = float(coordinates.horizontal) - (-84.92)
            #difv = float(coordinates.vertical) - (-1.6)

            moveh = int(round(difh / 0.0176,0))
            movev = int(round(difv / 0.0176,0))

            lrmoveh=int(round(moveh*0.3))
            lrmovev=int(round(movev*0.3))

            print(moveh)
            keys.directMouse(lrmoveh, -lrmovev)
            time.sleep(0.004)

            print(coordinates.X, coordinates.Y, coordinates.Z, coordinates.horizontal, coordinates.vertical,
                  coordinates.keyspressed, "PREDICTED:", round(predicted,2),"DIFh:",round(difh,2),"DIFv",round(difv,2))


        except:
            pass


time.sleep(3)
for cnt in range(100):
    # Get amount of files
    path, dirs, files = next(os.walk(r"C:\Users\emill\PycharmProjects\OWcheaters/Cheaters"))
    file_count = len(files)
    import pandas as pd

    # Clear the log file to avoid it becoming too large (maybe slower to read if file is bigger?)
    with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log",
              "w") as f:
        f.close()
    now = time.time()
    print(now)

    x = "go"
    while x != "stop":
        x = getpos(file_count)
    end = time.time()
    duration = end - now
    print(duration)
    df = pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/Cheaters/cords{file_count}.csv")
    print(len(df))
    print(len(df) / duration)
    time.sleep(1)