import time
import pyautogui
from tset import key_check
import csv
import os
from testing import Keys
import pickle
keys = Keys()

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelv.model", "rb") as f:
    cb_modelv = pickle.load(f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelh.model", "rb") as f:
    cb_modelh = pickle.load(f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelc.model", "rb") as f:
    cb_modelc = pickle.load(f)


def getpos(filenumber):
    keys.directKey("p")
    time.sleep(0.01)
    keys.directKey("p", keys.key_release)
    start = time.time()
    try:
        with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log",
                  "r") as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            splitted = last_line.split(";")

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

            if "O" in coordinates.keyspressed:
                return "O"

            predictedh = cb_modelh.predict(
                [float(coordinates.X), float(coordinates.Y), float(coordinates.Z)])
            difh=float(coordinates.horizontal)-predictedh


            predictedv = cb_modelv.predict(
                                [float(coordinates.X), float(coordinates.Y), float(coordinates.Z)])
            difv = float(coordinates.vertical)-predictedv


            moveh = int(round(difh / 0.0176,0))
            movev = int(round(difv / 0.0176,0))

            lrmoveh=int(round(moveh*1))
            lrmovev=int(round(movev*1))


            keys.directMouse(lrmoveh, -lrmovev)
            time.sleep(0.0004)

            keyp=cb_modelc.predict([float(coordinates.X), float(coordinates.Y), float(coordinates.Z),float(coordinates.horizontal),float(coordinates.vertical)])

            if keyp==0:
                keyp="A"
            elif keyp==1:
                keyp="W"
            else: keyp="D"

            keys.directKey("A", keys.key_release)
            keys.directKey("W", keys.key_release)
            keys.directKey("D", keys.key_release)
            keys.directKey(keyp)
            time.sleep(0.001)


            #print(coordinates.X, coordinates.Y, coordinates.Z, coordinates.horizontal, coordinates.vertical,
            #      coordinates.keyspressed, "PREDICTEDh:", round(predictedh,2),"PREDICTEDv:", round(predictedv,2),"DIFh:",round(difh,2),"DIFv",round(difv,2),"You should press",keyp)
            print(f"Position (X:{float(coordinates.X):.2f} Y:{float(coordinates.Y):.2f} Z: {float(coordinates.Z):.2f})      Moved {difh:.2f} horizontally and {difv:.2f} vertically. Button pressed: {keyp}")
    except:
        pass




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
time.sleep(7)
x = "go"
for game in range(1000):
    x = getpos(file_count)
    if x=="O":
        exit()
    end = time.time()
    duration = end - now
    """df = pd.read_csv(f"C:/Users/emill/PycharmProjects/OWcheaters/Cheaters/cords{file_count}.csv")
    print(len(df))
    print(len(df) / duration)
    time.sleep(1)"""