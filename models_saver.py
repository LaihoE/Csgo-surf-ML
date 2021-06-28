import pandas as pd
bigboi = pd.read_csv("C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords1.csv")
for i in range(200,249):
    df2=pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords{i}.csv")
    bigboi = pd.concat([bigboi, df2], ignore_index=True)

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pickle

df=bigboi
"""listaX=[]
listaY=[]
listaZ=[]

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
"""


def generate_time_lags(df, n_lags):
    df_n = df.copy()
    cols = ["X","Y","Z"]
    for c in cols:
        for n in range(1, n_lags + 1):
            df_n[f"{c}_lag{n}"] = df_n[f"{c}"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


input_dim = 100

df = generate_time_lags(df, input_dim)
print(df)


# GET OUT THE "P" THAT IS THE KEY FOR GETTING YOUR POSITION (NOT INTERESTED)

bigkeys=[]
for x in df["keys"]:
    keys = []
    if x[2] != "P":
        keys.append(x[2])
    try:
        if x[7] != "P":
            keys.append(x[7])
    except:
        pass
    bigkeys.append(keys)

df["newkeys"]=bigkeys

# If keys list is empty
biglist=[]
for i in df["newkeys"]:
    if len(i)>0:
        output=i[0]
    else:
        output="A"
    biglist.append(output)
df["newkeys"] = biglist
print(df)

df = df.drop("keys", axis=1)
cleandf=df

df["newkeys"] = df["newkeys"].replace("A", 0)
df["newkeys"] = df["newkeys"].replace("W", 1)
df["newkeys"] = df["newkeys"].replace("D", 2)
df["newkeys"] = df["newkeys"].replace("F", 3)


df=df.drop("newkeys",axis=1)
df=df.drop("vertical",axis=1)
X = df.drop('horizontal', axis=1)
y = df['horizontal']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
print(df)


cb_modelh = CatBoostRegressor(iterations=500,
                              depth=8,
                              task_type="CPU",
                              eval_metric='MAE',
                              random_seed=42,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=50,
                              od_wait=10,
                              learning_rate=0.0555
                              )
cb_modelh.fit(X_train, y_train, verbose=50)

"""random_search = RandomizedSearchCV(cb_model, param_distributions=params,verbose=5)
random_search.fit(X_train,y_train,verbose=0)
print(random_search.best_params_)
print(random_search.best_score_)"""



df=cleandf
df["newkeys"] = df["newkeys"].replace("A", 0)
df["newkeys"] = df["newkeys"].replace("W", 1)
df["newkeys"] = df["newkeys"].replace("D", 2)
df["newkeys"] = df["newkeys"].replace("F", 3)


df=df.drop("newkeys",axis=1)
df=df.drop("horizontal",axis=1)
X = df.drop('vertical', axis=1)
y = df['vertical']
print("DF BEFORE vert")
print(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


cb_modelv = CatBoostRegressor(iterations=500,
                              depth=8,
                              task_type="CPU",
                              eval_metric='R2',
                              random_seed=42,
                              bagging_temperature=0.2,
                              od_type='Iter',
                              metric_period=50,
                              od_wait=10,
                              learning_rate=0.0555
                              )
cb_modelv.fit(X_train, y_train, verbose=50)

df=cleandf
df["newkeys"]=df["newkeys"].replace("A",0)
df["newkeys"]=df["newkeys"].replace("W",1)
df["newkeys"]=df["newkeys"].replace("D",2)
df["newkeys"]=df["newkeys"].replace("F",42)
df["newkeys"]=df["newkeys"].replace("Y",42)
df["newkeys"]=df["newkeys"].replace("S",42)
df["newkeys"]=df["newkeys"].replace("Q",42)
df["newkeys"]=df["newkeys"].replace(" ",42)
df["newkeys"]=df["newkeys"].replace(3,42)
df["newkeys"]=df["newkeys"].replace("3",42)
df["newkeys"]=df["newkeys"].replace("O",42)
df["newkeys"]=df["newkeys"].replace("G",42)
df["newkeys"]=df["newkeys"].replace("I",42)
df["newkeys"]=df["newkeys"].replace("B",42)

#df=df[~df.newkeys.str.contains("QWERTYUIOPASDFGHJKLZXCVBNM3456789")]

print(df)

X=df.drop('newkeys',axis=1)
y=df['newkeys']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01,random_state=42)

# Check if illegal values (0-2 only wanted)
for i in y:
    if i != 0:
        if i != 1:
            if i != 2:
                if i != 42:
                    print(i)


cb_modelc = CatBoostClassifier(iterations=1000,
                             depth=8,
                             task_type="CPU",
                             eval_metric='Accuracy',
                             random_seed=42,
                             bagging_temperature=0.2,
                             od_type='Iter',
                             metric_period=50,
                             od_wait=10,
                             learning_rate=0.0555
                             )
cb_modelc.fit(X_train, y_train, verbose=50)

# save models

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelv.model", "w+b") as f:
    pickle.dump(cb_modelv, f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelh.model", "w+b") as f:
    pickle.dump(cb_modelh, f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelc.model", "w+b") as f:
    pickle.dump(cb_modelc, f)