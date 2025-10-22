from env import ContinuousBuildingControlEnvironment as BEnv
import numpy as np
import os
from utils import sample_param, custom_plot
from ddpg_online import ddpg_online
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    #set up environment
    data_file ='weather_data_2013_to_2017_winter_pandas.csv'
    policy_file = "model/best/saved_model_999.ckpt"
    online_start = 6000.
    online_end = 8160.
    env = BEnv(data_file, start = online_start, end = online_end,
               C_env = 3.1996e6,
               C_air = 3.5187e5,
               R_rc = 0.00706,
               R_oe = 0.02707,
               R_er = 0.00369)
    
	#run ddpg
    T_air, time, T_out, Q_SG, action_list, energy_list, penalty_list, \
    temp_metric_list, lb_list, ub_list = \
    ddpg_online(env = env, env_idx = 0, policy_file = policy_file,
                start = online_start, end = online_end,
                gamma = 0.99, epochs = 100,
                pi_lr = 1e-4, q_lr = 1e-3,
                hidden_sizes = (64, 64, 64, 64), activation = tf.nn.relu,
                max_ep_len = 4320, save_freq = 5, steps_per_epoch = 4320,
                replay_size = int(1e8), polyak = 0.995, batch_size = 100,
                update_after = 6, update_every = 12,
                act_noise = 0.002,
                rand_act_ratio = 0.)
    
    #custom_plot(T_air[2100:2300], time[2100:2300], T_out[2100:2300], Q_SG[2100:2300], 
    #            action_list[2100:2300], energy_list[2100:2300], penalty_list[2100:2300], 
    #            lb_list[2100:2300], ub_list[2100:2300], 9999, "meta")
    
    print(f"Energy Use in kWh: {float(energy_list[-1]):.2f}")
	print(f"# of Hours out of Bounds: {float(penalty_list[-1]):.0f}")
	print(f"Temperature Exceedance in degC-hr: {float(temp_metric_list[-1]):.2f}")
	
	# Handle multi-dimensional action array gracefully
	if isinstance(action_list, np.ndarray) and action_list.ndim > 1:
	    df_action = pd.DataFrame(action_list, columns=['SAT_sp', 'ZAT_sp'])
	else:
	    df_action = pd.DataFrame({'action': action_list})
	
	# Combine outputs
	df = pd.DataFrame({
	    'energy_meta': energy_list.flatten(),
	    'penalty_meta': penalty_list,
	    'exceedance_meta': temp_metric_list
	})
	df = pd.concat([df, df_action], axis=1)
	
	df.to_csv('results_meta.csv', index=False)

if __name__ == "__main__":
    main()
