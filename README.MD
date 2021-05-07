# Using Machine Learning to Surf
Video of the bot in action  
https://www.youtube.com/watch?v=v_V0mErWGpk&ab_channel=EmilLaiho  
Currently the model is just given positional data (x,y,z and camera angle) and keys pressed  
Model used is Catboost,similar to XGBoost,(not "deep-learning"). 3 different models are used to predict 
how much the camera angle should be rotated horizontally, veritcally and what key to press.