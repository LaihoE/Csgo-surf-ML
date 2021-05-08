from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
from gym import spaces
import time

def mainloop():
    # THE FOLLOWING COMMAND HAS TO BE PUT INTO CONSOLE. It starts logging your console

    # con_logfile console.log

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
                #keyspressed = key_check()
            return coordinates
            # SAFETY key will stop the bot
    except:
        pass
# Clear the log file to avoid it becoming too large (maybe slower to read if file is bigger?)
with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log","w") as f:
    f.close()

def get_state():
    try:
        cords=mainloop()
        x=cords.X
        z=cords.Z
        y=cords.Y
        vertical=cords.vertical
        horizontal=cords.horizontal
        return np.array([x, z, y, vertical, horizontal])
    except:
        pass


class surfenv(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([0, 0, 0, -180, -180]), high=np.array([1, 1, 1, 180, 180]))
        self.observation_space = Box(low=np.array([-1000, -1500, -9000, -180, -90]),
                                     high=np.array([1000, 1500, 11000, 180, 90]))
        self.state = np.array([0, -500, 10208, 89, 90.44])
        self.surf_length = 1000

    def step(self, action):
        keys.directKey("p", keys.key_release)

        if action[0] == 1:
            keys.directKey("A")
        if action[1] == 1:
            keys.directKey("W")
        if action[2] == 1:
            keys.directKey("D")

        # Default to not moving angle
        move_v, move_g = 0, 0

        if action[3] != 0:
            current_angle = get_state()
            current_v = float(current_angle[3])
            wanted_v = float(action[3])
            dif_v = current_v - wanted_v
            move_v = int(round(dif_v / 0.0176, 0))

        if action[4] != 0:
            current_angle = get_state()
            current_h = float(current_angle[4])
            wanted_h = float(action[4])
            dif_h = current_h - wanted_h
            move_h = int(round(dif_h / 0.0176, 0))

        keys.directMouse(move_h, -move_v)
        time.sleep(0.0004)

        self.state = get_state()

        if self.surf_length <= 0 or float(self.state[1]) > -100:
            done = True
        else:
            done = False

        if not (self.state is None):
            print("VICTORY")
            print(self.state)
            if float(self.state[1]) > -300:
                reward + 1
            else:
                reward - 1
            info = {}
        else:
            print("megafail")

        return self.state, reward, done, info

    def reset(self):
        # Teleport
        keys.directKey("p")
        time.sleep(0.02)
        keys.directKey("p", keys.key_release)

        keys.directKey("k")
        time.sleep(0.02)
        keys.directKey("k", keys.key_release)

        # Rotate camera to default (90,0)
        current_angle = get_state()

        current_h = float(current_angle[4])
        current_v = float(current_angle[3])

        wanted_h = 90
        wanted_v = 0

        dif_h = current_h - wanted_h
        dif_v = current_v - wanted_v

        # Sensitivity dependent
        move_h = int(round(dif_h / 0.0176, 0))
        move_v = int(round(dif_v / 0.0176, 0))

        keys.directMouse(move_h, -move_v)
        time.sleep(0.0004)

        self.state = get_state()

        return self.state