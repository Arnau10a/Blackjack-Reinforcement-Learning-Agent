from abstractagent import AbstractAgent
from blackjack import BlackjackObservation, BlackjackEnv, BlackjackAction
from typing import Dict, Tuple, List
import numpy as np

class SarsaAgent(AbstractAgent):
    """
    Implementation of an agent that learns utility estimates using SARSA algorithm.
    """

    def __init__(self, env: BlackjackEnv, number_of_episodes: int, alpha: float = 0.1, gamma: float = 1.0, epsilon: float = 0.1):
        self.env = env
        self.number_of_episodes = number_of_episodes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.Q: Dict[Tuple[int, int, int], float] = {}  # Q-values for (player_total, dealer_card, action)
        self.state_space = self._generate_state_space()

    def _generate_state_space(self) -> List[Tuple[int, int]]:
        """
        Generates the state space based on the defined space S1.
        """
        state_space = []
        for player_total in range(2, 22):
            for dealer_card in range(1, 12):
                state_space.append((player_total, dealer_card))
        return state_space

    def train(self):
        for player_hand_value, dealer_card in self.state_space:
            for action in [BlackjackAction.HIT.value, BlackjackAction.STAND.value]:
                self.Q[(player_hand_value, dealer_card, action)] = 0

        for i in range(self.number_of_episodes):
            observation, _ = self.env.reset()
            terminal = False
            reward = 0
            action = self._epsilon_greedy_action(observation) # Choose action using epsilon-greedy policy
            while not terminal:
                next_observation, reward, terminal, _, _ = self.env.step(action)
                next_action = self._epsilon_greedy_action(next_observation)
                # SARSA update
                self.Q[(observation.player_hand.value(), observation.dealer_hand.value(), action)] += \
                    self.alpha * (reward + self.gamma * self.Q.get((next_observation.player_hand.value(), next_observation.dealer_hand.value(), next_action), 0) -
                                self.Q.get((observation.player_hand.value(), observation.dealer_hand.value(), action), 0))
                observation = next_observation
                action = next_action


    def _epsilon_greedy_action(self, observation: BlackjackObservation) -> int:
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice([BlackjackAction.HIT.value, BlackjackAction.STAND.value])
        else:
            # Choose action with maximum Q-value
            player_hand_value = observation.player_hand.value()
            dealer_card = observation.dealer_hand.value()
            hit_value = self.Q.get((player_hand_value, dealer_card, BlackjackAction.HIT.value), 0)
            stand_value = self.Q.get((player_hand_value, dealer_card, BlackjackAction.STAND.value), 0)
            return BlackjackAction.HIT.value if hit_value > stand_value else BlackjackAction.STAND.value

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool) -> float:
        """
        Returns the learned Q-value for the given observation.
        """
        player_hand_value = observation.player_hand.value()
        dealer_card = observation.dealer_hand.value()
        return max(self.Q.get((player_hand_value, dealer_card, BlackjackAction.HIT.value), 0),
                   self.Q.get((player_hand_value, dealer_card, BlackjackAction.STAND.value), 0))

