import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Optional
import numpy as np


'''
This file is based on:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/roulette.py
The modification is done so that some common mistakes in implementation
are more likely to arise.
'''

INIT_MAX = 10
MAX_ALLOWED_TOKENS = 20


class SMUTutorialEnv(gym.Env):
    """Simple roulette environment

    The roulette wheel has 37 spots. If the bet is 0 and a 0 comes up,
    you win a reward of 35. If the parity of your bet matches the parity
    of the spin, you win 1. Otherwise, you lose -1.

    The long run reward for playing 0 should be -1/37 for any state
    The last action (37) stops the rollout for a return of your reward so far.

    You start with a random number of tokens which you bet. Initially, it is between
    one and 10. In each turn, you bet one token.

    The observation contains two parts: first is your cumulative reward so
    far, the second is the last number that has fallen on the roulette.

    Action 37 means that you want to cash your reward and walk away. The
    casino does not allow you to play on loan. Also if you own more than
    20 tokens, the security expells you from the casino.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False, spots=37, name=""):
        self.n = spots + 1
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.MultiDiscrete(np.array([MAX_ALLOWED_TOKENS *2, self.n]))
        self.seed()
        self.tokens:int = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        if action == self.n - 1:
            # observation, reward, done, info
            return np.array([self.tokens, self.get_val()]), self.tokens, True, False, {}

        # N.B. np.random.randint draws from [A, B) while random.randint draws from [A,B]
        val = self.get_val()
        if val == action == 0:
            self.tokens += self.n - 3
        elif val != 0 and action != 0 and val % 2 == action % 2:
            self.tokens += 1
        else:
            self.tokens += -1

        if self.tokens > MAX_ALLOWED_TOKENS or self.tokens < 1:
            return np.array([self.tokens, val]), self.tokens, True, False, {}

        return np.array([self.tokens, val]), 0, False, False, {}

    def reset(self, **kwargs):
        self.tokens = np.random.randint(1, INIT_MAX)
        return np.array([self.tokens, self.get_val()]), {}

    def render(self, mode='human', close=False):
        """
        Prints the situation to the terminal. Call env.render() any time
        you need to see the game. This method is useful for debugging.
        """
        print("You own " + str(self.tokens) + " tokens.")

    def get_val(self):
        return np.random.randint(0, self.n - 1)

