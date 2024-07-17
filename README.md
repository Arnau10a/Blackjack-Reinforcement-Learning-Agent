# Simplified Blackjack Reinforcement Learning Agent

## Project Overview
This project involves implementing an agent capable of playing a simplified version of the blackjack game using reinforcement learning techniques. The primary goal is for the agent to score more than the dealer without exceeding a total of 21.

## Rules of the Game
- Standard 52-card deck.
- The player is dealt two cards and sees one of the dealer's cards.
- The player can choose to draw another card or stop.
- The dealer draws cards until their total is at least 17.
- Face cards (Jack, Queen, King) are worth 10 points, and Aces can be 1 or 11.
- The player loses if their total exceeds 21, even if the dealer also busts.
- The winner is the one with the higher total without busting; ties occur with equal totals.

## Implementation Details
- **Environment**: Uses the Gymnasium library (formerly OpenAI Gym).
- **Files**:
  - `blackjack.py`: Environment implementation.
  - `carddeck.py`: Card, deck, and hand models.
  - `main.py`: Contains main logic (do not modify code above specified comments).
  - `randomagent.py`: Dummy agent making random decisions.
  - `dealeragent.py`: Agent following dealer's fixed strategy.
  - `tdagent.py`: Implement a passive reinforcement learning agent using temporal difference.
  - `sarsaagent.py`: Implement a SARSA algorithm.
  - `evaluation.py`: Ideas for comparing different agents.
