from abstractagent import AbstractAgent
from dealeragent import DealerAgent
from evaluate import *
import gymnasium as gym
from gymnasium import wrappers
from gymnasium.envs.registration import register
from randomagent import RandomAgent
from sarsaagent import SarsaAgent
from tdagent import TDAgent

#import warnings
#warnings.filterwarnings("ignore", message="`np.bool8` is a deprecated alias for `np.bool_`")


def get_env() -> gym.Wrapper:
    """
    Creates the environment. Check the OpenAI Gym documentation.

    :rtype: Environment of the blackjack game that follows the OpenAI Gym API.
    """
    environment = gym.make('smu-blackjack-v2', render_mode="ansi")
    return wrappers.RecordEpisodeStatistics(environment, deque_size=100000000)


if __name__ == "__main__":
    # Registers the environment so that it can be used
    register(
        id='smu-blackjack-v2',
        entry_point='blackjack:BlackjackEnv'
    )
    # ######################################################
    # IMPORTANT: do not modify the code above this line! ###
    # ######################################################

    # here you can play with the code yourself
    # for example you may want to split the code to two phases - training and testing
    # or you may want to compare two agents
    # feel free to modify the number of games played (highly recommended!)
    # ... or whatever

    env = get_env()
    number_of_episodes = 100000  # TODO do not forget to change the number of episodes

    #agent: AbstractAgent = RandomAgent(env, number_of_episodes)
    #agent: AbstractAgent = DealerAgent(env, number_of_episodes)
    #agent: AbstractAgent = TDAgent(env, number_of_episodes)
    agent: AbstractAgent = SarsaAgent(env, number_of_episodes)
    # agent: AbstractAgent = AdvancedAgent(env, number_of_episodes)
    agent.train()

    # in evaluate.py are some ideas that you might want to use to evaluate the agent
    # feel free to modify the code as you want to
    evaluate(list(env.return_queue))
