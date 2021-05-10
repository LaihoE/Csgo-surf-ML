import pandas as pd
bigboi = pd.read_csv("C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords1.csv")
for i in range(50, 226):
    df2=pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/longrun/cords{i}.csv")
    bigboi = pd.concat([bigboi, df2], ignore_index=True)
print(bigboi)


df=bigboi
listaX=[]
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

print(df)