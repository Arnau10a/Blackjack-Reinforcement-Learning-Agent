from abstractagent import AbstractAgent
from blackjack import BlackjackObservation, BlackjackEnv, BlackjackAction
from carddeck import *
from typing import Dict


class TDAgent(AbstractAgent):
    """
    Implementation of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    This agent learns utility estimates using the temporal difference (TD) method.
    """
    def __init__(self, env: BlackjackEnv, number_of_episodes: int, alpha: float = 0.8, gamma: float = 0.9):
        """
        Initializes the agent.
        :param env: The environment in which the agent plays Blackjack.
        :param number_of_episodes: Number of episodes to train on.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        """
        super().__init__(env, number_of_episodes)
        self.alpha = alpha
        self.gamma = gamma
        self.state_values: Dict[BlackjackObservation, float] = {}

    def train(self):
        for _ in range(self.number_of_episodes):
            observation, _ = self.env.reset()
            terminal = False
            while not terminal:
                action = self.receive_observation_and_get_action(observation, terminal)
                next_observation, reward, terminal, _, _ = self.env.step(action)

                # Update state value estimate
                current_value = self.get_hypothesis(observation, terminal) # U(s)
                next_value = self.get_hypothesis(next_observation, terminal) # U(s')
                td_target = reward + self.gamma * next_value # R + gamma * U(s')
                td_error = td_target - current_value # R + gamma * U(s') - U(s)
                self.state_values[observation] = current_value + self.alpha * td_error # U(s) = U(s) + alpha * (R + gamma * U(s') - U(s))

                observation = next_observation

    def receive_observation_and_get_action(self, observation: BlackjackObservation, terminal: bool) -> int:
        return BlackjackAction.HIT.value if observation.player_hand.value() < 17 else BlackjackAction.STAND.value

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool) -> float:
        """
        Return the learned U value for the given observation.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :return: The learned U-value for the given observation.
        """
        if terminal:
            return 0.0  # Terminal states have zero value
        return self.state_values.get(observation, 0.0)