import csv
import pyautogui
import os
import time
from tset import key_check

# con_logfile console.log
# sv_accelerate 10
# sv_airaccelerate 800
#
#


def create_headers(filenumber):
    with open(f"C:/Users/emill/PycharmProjects/OWcheaters/Cheaters/cords{filenumber}.csv", "w", newline="\n")as f:
        thewriter = csv.writer(f)
        thewriter.writerow(["X", "Y", "Z",
                            "horizontal", "vertical"])

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
            print(coordinates.X, coordinates.Y, coordinates.Z,coordinates.horizontal, coordinates.vertical,coordinates.keyspressed)



            if float(coordinates.X) <= 80 and float(coordinates.Z) < -700:
                print("yes")
                return "stop"
            with open(f"C:/Users/emill/PycharmProjects/OWcheaters/Cheaters/cords{filenumber}.csv","a",newline="\n")as f:
                thewriter=csv.writer(f)
                thewriter.writerow([coordinates.X,coordinates.Y,coordinates.Z,
                                    coordinates.horizontal,coordinates.vertical,coordinates.keyspressed])
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
    create_headers(file_count)
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