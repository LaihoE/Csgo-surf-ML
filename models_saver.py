import pandas as pd
bigboi = pd.read_csv("C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords1.csv")
for i in range(60, 106):
    df2=pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords{i}.csv")
    bigboi = pd.concat([bigboi, df2], ignore_index=True)

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pickle

# GET OUT THE "P" THAT IS THE KEY FOR GETTING YOUR POSITION (NOT INTERESTED)
df=bigboi
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


cb_modelh = CatBoostRegressor(iterations=2000,
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


cb_modelv = CatBoostRegressor(iterations=1000,
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
cb_modelv.fit(X_train, y_train, verbose=0)

df=cleandf
df["newkeys"]=df["newkeys"].replace("A",0)
df["newkeys"]=df["newkeys"].replace("W",1)
df["newkeys"]=df["newkeys"].replace("D",2)
df["newkeys"]=df["newkeys"].replace("F",42)
df["newkeys"]=df["newkeys"].replace("Y",42)
df["newkeys"]=df["newkeys"].replace("S",42)
df["newkeys"]=df["newkeys"].replace("Q",42)
df["newkeys"]=df["newkeys"].replace(" ",42)
print(df)

X=df.drop('newkeys',axis=1)
y=df['newkeys']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01,random_state=42)

# Check if illegal values (0-2 only wanted)
for i in y:
    if i != 0:
        if i != 1:
            if i != 2:
                print(i)



cb_modelc = CatBoostClassifier(iterations=1000,
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
cb_modelc.fit(X_train, y_train, verbose=0)

# save models

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelv.model", "w+b") as f:
    pickle.dump(cb_modelv, f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelh.model", "w+b") as f:
    pickle.dump(cb_modelh, f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelc.model", "w+b") as f:
    pickle.dump(cb_modelc, f)