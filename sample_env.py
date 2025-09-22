from env import ContinuousBuildingControlEnvironmentTrain as BEnvTrain
from env import ContinuousBuildingControlEnvironmentTest as BEnvTest
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils import sample_param

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_file ='weather_data_2013_to_2017_winter_pandas.csv'

# Generate the samples
num_env_sample = 100
param = sample_param(num_env_sample)
print(param)
df = pd.DataFrame({'C_env': param[:, 0], 
                   'C_air':param[:, 1], 
				   'R_rc':param[:, 2], 
				   'R_oe':param[:, 3], 
df.to_csv('env_param.csv', index = False)
param = pd.read_csv('env_param.csv')

# Lower bound of the observation space
low = np.array([10.0, 18.0, 21.0, -40.0, 0., 50., 0.])
# Upper bound of the observation space
high = np.array([35.0, 27.0, 23.0, 40.0, 1100., 180., 23.])

# Test the env quality with 1 year winter data
start = 0.
end = 2000.

# Generate random actions
a = np.random.uniform(low=0.0, high=1.0, size = int(end-start)*2)

T_air = np.array([])
env_true = BEnvTest(data_file, start = start, end = end)

obs = env_true.reset()
for i in range(int(end-start)*2):
    obs, r, done, _, _, _ = env_true.step(a[i])
    s_t = obs * (high - low) + low
    T_air = np.append(T_air, s_t[1])

RMSE = np.array([])
for sample in range(len(param)):
    env_false = BEnvTrain(data_file, start = start, end = end, 
                          C_env = param.loc[sample, 'C_env'], 
                          C_air = param.loc[sample, 'C_air'], 
                          R_rc = param.loc[sample, 'R_rc'], 
                          R_oe = param.loc[sample, 'R_oe'], 
                          R_er = param.loc[sample, 'R_er'])
    
    T_air_false = np.array([])
    obs = env_false.reset()
    for t in range(int(end-start)*2):
        obs, r, done, _ = env_false.step(a[t])
        s_t = obs * (high - low) + low
        T_air_false = np.append(T_air_false, s_t[1])
    
    RMSE_sample = np.mean(np.sqrt((T_air_false - T_air)**2.))
    RMSE = np.append(RMSE, RMSE_sample)
    
    print(sample)

param['RMSE'] = RMSE
param.to_csv('env_param.csv', index = False)