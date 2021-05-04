import csv
import pyautogui
import os
import time
# con_logfile filename.txt
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
            print(coordinates.X, coordinates.Y, coordinates.Z,coordinates.horizontal, coordinates.vertical)
            with open(f"C:/Users/emill/PycharmProjects/OWcheaters/Cheaters/cords{filenumber}.csv","a",newline="\n")as f:
                thewriter=csv.writer(f)
                thewriter.writerow([coordinates.X,coordinates.Y,coordinates.Z,
                                    coordinates.horizontal,coordinates.vertical])
        except:
            pass

time.sleep(3)
# Get amount of files
path, dirs, files = next(os.walk(r"C:\Users\emill\PycharmProjects\OWcheaters/Cheaters"))
file_count = len(files)

# Clear the log file to avoid it becoming too large (maybe slower to read if file is bigger?)
with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log", "w") as f:
    f.close()
create_headers(file_count)
while True:
    getpos(file_count)
