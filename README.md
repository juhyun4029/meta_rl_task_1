# Meta-Learning Based Controller for Thermal Management

## Objective:
The aim is to develop an autonomous controller which provides personalised HVAC control for individual office occupant. 

We are focusing on using model-free RL algorithms to obtain the optimal control policy for a single office room in a building (the Ray W. Herrick Laboratory). We wish to extend this approach to a more generalized learning algorithm, possibly a meta-learning controller.

For achieving this objective, we need to do the following tasks:

 - We need to build a Gym-like environment. This will enable us to make multiple realizations of the universe which we are interested in.
 
 - We need to train & test various RL algorithms to obtain an agent capable of working on various realizations of the environment.
