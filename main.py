import time
from tset import key_check
from testing import Keys
import pickle
keys = Keys()

# Load in the 3 models
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelv.model", "rb") as f:
    cb_modelv = pickle.load(f)
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelh.model", "rb") as f:
    cb_modelh = pickle.load(f)
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelc.model", "rb") as f:
    cb_modelc = pickle.load(f)


def mainloop():
    # THE FOLLOWING COMMAND HAS TO BE PUT INTO CONSOLE. It starts logging your console

    # con_logfile console.log

    # P is binded ingame to showpos1 (it gives the positional data)
    keys.directKey("p")
    time.sleep(0.01)
    keys.directKey("p", keys.key_release)
    start = time.time()
    try:
        # Read console.log and get the relevant data
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
            # SAFETY key will stop the bot
            if "O" in coordinates.keyspressed:
                return "O"

            predicted_horiz_angle = cb_modelh.predict(
                [float(coordinates.X), float(coordinates.Y), float(coordinates.Z)])
            difh=float(coordinates.horizontal)-predicted_horiz_angle

            predicted_vertical_angle = cb_modelv.predict(
                                [float(coordinates.X), float(coordinates.Y), float(coordinates.Z)])
            difv = float(coordinates.vertical)-predicted_vertical_angle

            # User specific. Depends on your sens/dpi. Figured it out after some playing with it (0.8 in-game 800dpi)
            moveh = int(round(difh / 0.0176,0))
            movev = int(round(difv / 0.0176,0))

            # Can make it move towards the correct angle but not exactly. if constant is 1 then make cursor "teleport"
            # to the exact position it wants. Similar learning rate in deep learning. Seems to work well with 1
            lrmoveh=int(round(moveh*1))
            lrmovev=int(round(movev*1))

            # Move the mouse
            keys.directMouse(lrmoveh, -lrmovev)
            time.sleep(0.0004)

            # Predict what key to press (output is 0-2 corresponding to A,W,D)
            keyp = cb_modelc.predict([float(coordinates.X), float(coordinates.Y), float(coordinates.Z),float(coordinates.horizontal),float(coordinates.vertical)])

            if keyp==0:
                keyp="A"
            elif keyp==1:
                keyp="W"
            else: keyp="D"
            # Release the old loop holds
            keys.directKey("A", keys.key_release)
            keys.directKey("W", keys.key_release)
            keys.directKey("D", keys.key_release)
            # Press down key
            keys.directKey(keyp)
            time.sleep(0.001)
            print(f"Position (X:{float(coordinates.X):.2f} Y:{float(coordinates.Y):.2f} Z: {float(coordinates.Z):.2f})      Moved {difh:.2f} horizontally and {difv:.2f} vertically. Button pressed: {keyp}")
    except:
        pass

# Clear the log file to avoid it becoming too large (maybe slower to read if file is bigger?)
with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log","w") as f:
    f.close()

for game in range(1000):
    x = mainloop()
    if x=="O":
        exit()
