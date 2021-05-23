from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import csv
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
from gym import spaces
import time
from time import sleep

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
    listaX.append(float(coordinates.X))
    listaZ.append(float(coordinates.Z))
    return coordinates
        # SAFETY key will stop the bot

listaX=[]
listaZ=[]
def get_state():
    try:
        cords = mainloop()
        x = cords.X
        z = cords.Z
        y = cords.Y
        # vertical = cords.vertical
        # horizontal = cords.horizontal
        ret = np.array([x,z])

        return ret
    except:
        return 0

class surfenv(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = Box(low=np.array([-1000,-900]), high=np.array([1000,2000]), dtype=np.float32)
        self.state = get_state()
        self.surf_length = 200

    def step(self, action):

        if action == 0:
            keys.directKey("A")
        if action == 1:
            keys.directKey("W")
        if action == 2:
            keys.directKey("D")
        #if action == 3:
            #keys.directKey("S")

        returned_state = get_state()
        if returned_state != 0:
            self.state = returned_state
        print(returned_state)
        print(self.state)
        self.surf_length -= 1

        if self.surf_length <= 0 or float(self.state[1]) < -730 or float(self.state[0]) < -500 or float(self.state[0])> 500:
            done = True
        else:
            done = False


        if float(self.state[1]) > -500:
            reward = (float(self.state[1]) - 200) / 100
        else:
            reward = (float(self.state[1]) - 200) / 100

        print(action, reward)
        reward=0
        done=False



        keys.directKey("P", keys.key_release)
        keys.directKey("A", keys.key_release)
        keys.directKey("W", keys.key_release)
        keys.directKey("D", keys.key_release)
        keys.directKey("S", keys.key_release)
        info = {}
        return self.state, reward, done, info

    def reset(self):
        # Teleport
        keys.directKey("p")
        sleep(0.02)
        keys.directKey("p", keys.key_release)

        keys.directKey("k")
        sleep(0.05)
        keys.directKey("k", keys.key_release)

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
        self.surf_length = 150
        return self.state


import time
from keys import Keys
sleep(3)
keys = Keys()
env = surfenv()
states = env.observation_space.shape  # (1,)
actions = env.action_space.n  # 3
print(states)
print(states[0])
from time import time, sleep
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='relu'))

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
    memory = SequentialMemory(limit=50000, window_length=5)
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
                   nb_actions=actions, nb_steps_warmup=300, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
experimental_run_tf_function = False


sleep(4)
dqn.fit(env, nb_steps=5000, visualize=False, verbose=100)
print("TESTING")
print("TESTING")
print("TESTING")
print("TESTING")
sleep(5)
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
scores = dqn.test(env, nb_episodes=1, visualize=False)
#print(np.mean(scores.history['episode_reward']))
dqn.save_weights('dqn2_weights.h5f', overwrite=True)

print(listaX)

"""def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

env = surfenv()
actions = env.action_space.n
states = env.observation_space.shape

model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.load_weights('dqn_weights.h5f')
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
_ = dqn.test(env, nb_episodes=2, visualize=True)
"""
import matplotlib.pyplot as plt

plt.scatter(x=listaX,y=listaZ,s=2)
plt.show()