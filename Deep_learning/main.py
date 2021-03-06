import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tset import key_check
from testing import Keys
import pickle
keys = Keys()

# con_logfile console.log
# sv_accelerate 10
# sv_airaccelerate 800

# Load in the 3 models
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelv.model", "rb") as f:
    cb_modelv = pickle.load(f)
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelh.model", "rb") as f:
    cb_modelh = pickle.load(f)
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelc.model", "rb") as f:
    cb_modelc = pickle.load(f)

with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelv.model2", "rb") as f:
    cb_modelv2 = pickle.load(f)
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelh.model2", "rb") as f:
    cb_modelh2 = pickle.load(f)
with open("C:/Users/emill/PycharmProjects/OWcheaters/script/models/cb_modelc.model2", "rb") as f:
    cb_modelc2 = pickle.load(f)

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out


grumodel = torch.load("C:/data/GRU/AntiCheat4.pt",map_location=torch.device('cpu'))

def mainloop(lastX,lastY,lastZ):
    # THE FOLLOWING COMMAND HAS TO BE PUT INTO CONSOLE. It starts logging your console
    global bigdf
    # con_logfile console.log

    # P is binded ingame to showpos1 (it gives the positional data)
    keys.directKey("p")
    time.sleep(0.01)
    keys.directKey("p", keys.key_release)
    start = time.time()

    # Read console.log and get the relevant data
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
                SpeedX=float(X) - float(lastX)
                SpeedY=float(Y) - float(lastY)
                SpeedZ=float(Z) - float(lastZ)

            # SAFETY key will stop the bot
            if "O" in coordinates.keyspressed:
                return "O","0","0"

            def generate_time_lags(df, n_lags):
                df_n = df.copy()
                cols = ["X", "Y", "Z"]
                for c in cols:
                    for n in range(1, n_lags + 1):
                        df_n[f"{c}_lag{n}"] = df_n[f"{c}"].shift(n)
                df_n = df_n.iloc[n_lags:]
                return df_n


            #print(df)pppppp
            #df = pd.DataFrame()
            row = pd.DataFrame({"X":float(coordinates.X), "Y":float(coordinates.Y), "Z":float(coordinates.Z),"horizontal":float(coordinates.horizontal),"vertical":float(coordinates.vertical),"SpeedX":coordinates.SpeedX,"SpeedY":coordinates.SpeedY,"SpeedZ":coordinates.SpeedZ},index=[0])
            bigdf = bigdf.append(row,ignore_index=True)

            input_dim = 100
            print(bigdf)
            if len(bigdf) > 100:
                X = bigdf.iloc[-100:].values # np array
                print(X.shape)
                predicted_horiz_angle = cb_modelh2.predict(
                    [float(coordinates.X), float(coordinates.Y), float(coordinates.Z), coordinates.SpeedX,
                     coordinates.SpeedY, coordinates.SpeedZ])
                difh = float(coordinates.horizontal) - predicted_horiz_angle
                print("DIFH", difh)
                predicted_vertical_angle = cb_modelv2.predict(
                    [float(coordinates.X), float(coordinates.Y), float(coordinates.Z), coordinates.SpeedX,
                     coordinates.SpeedY, coordinates.SpeedZ])
                difv = float(coordinates.vertical) - predicted_vertical_angle








                # User specific. Depends on your sens/dpi. Figured it out after some playing with it (0.8 in-game 800dpi)
                moveh = int(round(difh / 0.0176, 0))
                movev = int(round(difv / 0.0176, 0))

                # Can make it move towards the correct angle but not exactly. if constant is 1 then make cursor "teleport"
                # to the exact position it wants.
                lrmoveh = int(round(moveh * 0.8))
                lrmovev = int(round(movev * 0.8))

                # Move the mouse
                keys.directMouse(lrmoveh, -lrmovev)
                time.sleep(0.0004)

                # Predict what key to press (output is 0-2 corresponding to A,W,D)
                keyp = cb_modelc2.predict(
                    [float(coordinates.X), float(coordinates.Y), float(coordinates.Z), float(coordinates.horizontal),
                     float(coordinates.vertical), coordinates.SpeedX, coordinates.SpeedY, coordinates.SpeedZ])
                X = X.reshape(-1,100,8)
                X = torch.tensor(X)
                X = X.float()
                prd = grumodel.forward(X)
                print("GRU",prd)
                pp = torch.argmax(prd)
                print(pp.item())
                keyp = pp.item()

                if keyp == 0:
                    keyp = "A"
                elif keyp == 1:
                    keyp = "W"
                else:
                    keyp = "D"
                # Release the old loop holds
                keys.directKey("A", keys.key_release)
                keys.directKey("W", keys.key_release)
                keys.directKey("D", keys.key_release)
                # Prespppps down key
                keys.directKey(keyp)
                time.sleep(0.001)
                print(
                    f"Position (X:{float(coordinates.X):.2f} Y:{float(coordinates.Y):.2f} Z: {float(coordinates.Z):.2f})      Moved {difh:.2f} horizontally and {difv:.2f} vertically. Button pressed: {keyp}")
                return coordinates.X, coordinates.Y, coordinates.Z
            else:

                predicted_horiz_angle = cb_modelh2.predict(
                 [float(coordinates.X), float(coordinates.Y), float(coordinates.Z),coordinates.SpeedX,coordinates.SpeedY,coordinates.SpeedZ])
                difh = float(coordinates.horizontal) - predicted_horiz_angle
                print("DIFH", difh)
                predicted_vertical_angle = cb_modelv2.predict([float(coordinates.X), float(coordinates.Y), float(coordinates.Z),coordinates.SpeedX,coordinates.SpeedY,coordinates.SpeedZ])
                difv = float(coordinates.vertical) - predicted_vertical_angle

                # User specific. Depends on your sens/dpi. Figured it out after some playing with it (0.8 in-game 800dpi)
                moveh = int(round(difh / 0.0176, 0))
                movev = int(round(difv / 0.0176, 0))

                # Can make it move towards the correct angle but not exactly. if constant is 1 then make cursor "teleport"
                # to the exact position it wants. Similar learning rate in deep learning. Seems to work well with 1
                lrmoveh = int(round(moveh * 0.8))
                lrmovev = int(round(movev * 0.8))

                # Move the mouse
                keys.directMouse(lrmoveh, -lrmovev)
                time.sleep(0.0004)

                # Predict what key to press (output is 0-2 corresponding to A,W,D)
                keyp = cb_modelc2.predict([float(coordinates.X), float(coordinates.Y), float(coordinates.Z),float(coordinates.horizontal),float(coordinates.vertical),coordinates.SpeedX,coordinates.SpeedY,coordinates.SpeedZ])

                if keyp == 0:
                    keyp = "A"
                elif keyp == 1:
                    keyp = "W"
                else:
                    keyp = "D"
                # Release the old loop holds
                keys.directKey("A", keys.key_release)
                keys.directKey("W", keys.key_release)
                keys.directKey("D", keys.key_release)
                # Prespppps down key
                keys.directKey(keyp)
                time.sleep(0.001)
                print(
                    f"Position (X:{float(coordinates.X):.2f} Y:{float(coordinates.Y):.2f} Z: {float(coordinates.Z):.2f})      Moved {difh:.2f} horizontally and {difv:.2f} vertically. Button pressed: {keyp}")
                return coordinates.X, coordinates.Y, coordinates.Z

    except Exception as e:
        print(e)
# Clear the log file to avoid it becoming too large (maybe slower to read if file is bigger?)
with open(r"C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/console.log","w") as f:
    f.close()

time.sleep(2)
lastX,lastY,lastZ=0,0,0
bigdf = pd.DataFrame({"X":0, "Y":0, "Z":0,"horizontal":0,"vertical":0,"SpeedX":0,"SpeedY":0,"SpeedZ":0},index=[0])
for game in range(1000):
    try:
        lastX,lastY,lastZ = mainloop(lastX,lastY,lastZ)
    except:
        pass
    if lastX=="O":
        exit()