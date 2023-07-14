# Q-Learning

A low learning curve implementation of Q-learning.

[![PyPI](https://img.shields.io/pypi/v/qlearning)](https://pypi.org/project/Simple-Q/)
[![License](https://img.shields.io/pypi/l/qlearning)](https://github.com/Nesta-gitU/Simple-Q/blob/main/Simple_Q/LICENSE).

Q-learning is a reinforcement learning algorithm that allows an agent to learn an optimal policy for sequential decision-making problems. This implementation provides an easy-to-use and beginner-friendly approach to Q-learning, making it accessible to users with varying levels of experience. This implementation is build to work with "GYM" environments from the [Gymnasium Project](https://gymnasium.farama.org/).

## Features

- Provides flexibility to work with either integer or custom objects for states and actions.
- Offers customizable learning rate and exploration-exploitation trade-off parameters.
- Implements epsilon-greedy strategy for action selection.
- Supports polynomial and power decay functions for learning rate and exploration parameter respecitively. 
  The learning rate decay is based on the research in Even-Dar et al. (2003). When lambda is in (0,1) this implementation of Q-learning satisfies the conditions for probability one convergence in Watkins and Dayan (1992).

## Installation

Install the `Simple-Q` package from PyPI:

```shell
pip install Simple-Q
```

## Usage

```python
from Simple-Q import Qlearning

# Create the Q-learning agent
agent = Qlearning(states=10, actions = ['forward', 'left', 'right' 'stop'])

# Use the Q-learning agent in your reinforcement learning loop
state = env.reset()
N = 100
for n in range(N):
    action = agent.get_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.update_q_table(state, action, reward, next_state)
    state = next_state
    agent.update_epsilon()
```

See also the [Savings Problem](https://github.com/Nesta-gitU/Simple-Q/tree/main/SavingsProblem) example in the GitHub repository.

## Documentation

The Q-Learning class provides the following parameters:

| Parameter             | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `states`              | Number of states or a list of immutable objects representing states. If a list is passed, the states are mapped to integers starting from 0, and the immutable objects (not indices) are expected as input for the other methods. |
| `actions`             | Number of actions or a list of immutable objects representing actions. If a list is passed, the actions are mapped to integers starting from 0, and the immutable objects (not indices) are expected as input for the other methods. |
| `alpha`               | Learning rate parameter. If `'polynomial'`, a polynomial decay is used. If `int` or `float`, a fixed value is used. |
| `w`                   | Decay exponent for the polynomial decay. Only applicable when `alpha='polynomial'`. |
| `gamma`               | Discount factor for future rewards.                          |
| `epsilon_decay`       | Epsilon decay function. If `'power'`, a power decay is used. |
| `epsilon`             | Initial value for the exploration-exploitation trade-off parameter. |
| `epsilon_decay_factor`| Decay factor for epsilon. Determines how it decays over time. |
| `epsilon_min`         | Minimum value for epsilon.                                   |

The Q-Learning class provides the following attributes:

| Attribute  | Description                                                  |
|------------|--------------------------------------------------------------|
| `q_table`  | Q-table storing the learned Q-values for state-action pairs. |

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/Nesta-gitU/Simple-Q/blob/main/Simple_Q/LICENSE).

## Acknowledgements

This code is originally based upon the [Q-Learning implementation](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-learning-9.3.1.py) from the book: Advanced Deep Learning with TensorFlow 2 and Keras by Rowel Atienza. However many changes have been made to the original code, including the addition of the polynomial learning rate decay as well as the ability to work with custom objects for states and actions. 

## References
Even-Dar, E., Y. Mansour, and P. Bartlett (2003). Learning rates for q-learning. Journal of machine
learning Research 5(1).

Watkins, C. J. and P. Dayan (1992, May). Technical note: Q-learning. Machine Learning 8(3),
279–292
