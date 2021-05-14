from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

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

    # P is binded ingame to showpos1 (it gives the positional data)
    keys.directKey("p")
    sleep(0.002)
    keys.directKey("p", keys.key_release)
    # start = time.time()

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
            # keyspressed = key_check()

        return coordinates
        # SAFETY key will stop the bot


def get_state():
    cords = mainloop()
    x = cords.X
    z = cords.Z
    y = cords.Y
    # vertical = cords.vertical
    # horizontal = cords.horizontal
    ret = np.array([x])
    print(ret[0])
    return ret[0]


class surfenv(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = Box(low=np.array([-1000]), high=np.array([1000]), dtype=np.float32)
        self.state = 1470
        self.surf_length = 50

    def step(self, action):
        print("ACTION:", action)
        print("SELF_state", self.state)
        print("SURF_LEN:", self.surf_length)
        if action == 0:
            keys.directKey("A")
        if action == 1:
            keys.directKey("W")
        if action == 2:
            keys.directKey("D")
        if action == 3:
            keys.directKey("S")

        self.state = get_state()

        self.surf_length -= 1

        if self.surf_length <= 0 or float(self.state) > 1750:
            done = True
        else:
            done = False


        if float(self.state) > 1500:
            reward = float(self.state) - 1500
        else:
            reward = float(self.state) - 1500


        keys.directKey("P", keys.key_release)
        keys.directKey("A", keys.key_release)
        keys.directKey("W", keys.key_release)
        keys.directKey("D", keys.key_release)
        keys.directKey("S", keys.key_release)
        info = {}
        print("DONEEEE",done)
        return self.state, reward, done, info

    def reset(self):
        # Teleport
        keys.directKey("p")
        sleep(0.02)
        keys.directKey("p", keys.key_release)

        keys.directKey("k")
        sleep(0.5)
        keys.directKey("k", keys.key_release)

        # Rotate camera to default (90,0)
        """current_angle = get_state()

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
        sleep(0.0004)

        self.state = get_state()
        self.surf_length = 10
        self.state = get_state()"""
        reward=0
        self.state = get_state()
        self.surf_length = 50
        print(self.state)
        return self.state


import time
from keys import Keys
keys = Keys()

env = surfenv()
states = env.observation_space.shape  # (1,)
actions = env.action_space.n  # 3
print(states)

from time import time, sleep
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))  # states=(1,)
    # model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model




#del model

env = surfenv()

env.observation_space.sample()

#model.summary()


model = build_model(states, actions)

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=300, target_model_update=1e-2)
    return dqn

# WHY THE FUCK DOES THIS WORK?
dqn = build_agent(model, actions)
try:
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
except:
    print("wtf")

del model


env = surfenv()

env.observation_space.sample()

#model.summary()


model = build_model(states, actions)

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])





experimental_run_tf_function = False


sleep(4)
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

