from game import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import wrappers
import numpy as np


register(
    id='smu-rl-env-v1',
    entry_point='game:SMUTutorialEnv'
)
envsimple = gym.make('smu-rl-env-v1')

def policy(observation):
    if observation[1] > 27 or np.random.random() < 0.2:
        return 37
    return np.random.randint(0, 37)

alpha = 0.1

# TD method
number_of_epochs = 10 # TODO pick your own number and discount factor
discount_factor = 0.99

# this is the OpenAI Gym environment, you will access
env = envsimple
# TODO : here, store utility function estimates, i.e., a dictionary, an array or anything else
U = np.zeros(env.observation_space.nvec[0] * env.observation_space.nvec[1]) # TODO : initialize the utility function estimates
Us101 = np.zeros(number_of_epochs) 
maxDelta = np.zeros(number_of_epochs) 

for i in range(number_of_epochs):
    observation, _ = env.reset()
    print(observation)
    terminal = False

    while (not terminal):
        observation, reward, terminal, _, _ = env.step(policy(observation))
        print(observation, reward, terminal)
        
        U[state] += alpha * (reward + discount_factor * U[next_state] - U[state])
        state = next_state

        maxDelta[i] = max(maxDelta[i], abs(U[state] - U[next_state])) # TODO : store the maximum update to U

        # TODO :  store the value of a state (you can pick a different one if you want)
        Us101[i] = U[101]

import matplotlib.pyplot as plt

def plot_series(arr, fileName = None):
    plt.plot(arr)
    if fileName is not None:
        plt.savefig(fileName)

plot_series(Us101, "Us101.png")
# put a name of a file as a second parameter if you want to save the figure
    