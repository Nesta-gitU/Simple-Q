import numpy as np
from bidict import bidict

class Qlearning():
    """
    Q-learning algorithm for reinforcement learning.

    Parameters
    ----------
    states : int or list of immutable objects
        Number of states or a list of immutable objects representing states.
        if a list is passed, the states are mapped to integers starting from 0. 
        and the imutable objects, not indices are expected as input for the other methods.

    actions : int or list of immutable objects
        Number of actions or a list of immutable objects representing actions.
        if a list is passed, the actions are mapped to integers starting from 0.
        and the imutable objects, not indices are expected as input for the other methods.

    alpha : 'polynomial', int, float, default='polynomial'
        Learning rate parameter. If 'polynomial', a polynomial decay is used.
        If int or float, a fixed value is used.

    w : float, default=0.67
        Decay exponent for the polynomial decay. Only applicable when
        alpha='polynomial'.

    gamma : float, default=0.8
        Discount factor for future rewards.

    epsilon_decay : 'power', default='power'
        Epsilon decay function. If 'power', a power decay is used.

    epsilon : float, default=0.9
        Initial value for the exploration-exploitation trade-off parameter.

    epsilon_decay_factor : float, default=0.9
        Decay factor for epsilon. Determines how it decays over time.

    epsilon_min : float, default=0.1
        Minimum value for epsilon.

    Attributes
    ----------
    q_table : ndarray of shape (n_states, n_actions)
        Q-table storing the learned Q-values for state-action pairs.
    """

    _parameter_constraints: dict = {
        "alpha": ['polynomial'], #or int, float 
        "epsilon_decay": ['power'] 
    }

    _USE_STATE_DICT: bool = False
    _USE_ACTION_DICT: bool = False

    def __init__(self, 
                 states, #int or list of immutable objects
                 actions, #int or list of immutable objects
                 alpha = 'polynomial', #or int, float
                 w = 0.67, 
                 gamma = 0.8,
                 epsilon_decay = 'power',
                 epsilon = 0.9,
                 epsilon_decay_factor = 0.9,
                 epsilon_min = 0.1): 
        
        if isinstance(states, list):
            self.n_states: int = len(states)
            self._USE_STATE_DICT: bool = True
            self._state_dict: bidict = bidict({state: i for i, state in enumerate(states)})
        else:
            self.n_states: int = states
        
        if isinstance(actions, list):
            self.n_actions: int = len(actions)
            self._USE_ACTION_DICT: bool = True
            self._action_dict: bidict = bidict({action: i for i, action in enumerate(actions)})
        else:
            self.n_actions: int = actions

        self.alpha = alpha
        self.gamma: float = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon: float = epsilon
        self.epsilon_decay_factor: float = epsilon_decay_factor
        self.epsilon_min: float = epsilon_min

        self.q_table = np.full([self.n_states, self.n_actions], fill_value= -np.inf)
        
        
        if alpha == 'polynomial':
            self.n_visited = np.zeros([self.n_states, self.n_actions])
            self.w = w

        # warnings for invalid parameters
        if alpha not in self._parameter_constraints["alpha"] and not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be in {}, or of types: int, float".format(self._parameter_constraints["alpha"]))
        
        if isinstance(alpha, (int, float)):
            if alpha > 1 or alpha <= 0:
                raise Warning("alpha should be in (0,1]")
        
        if epsilon_decay not in self._parameter_constraints["epsilon_decay"]:
            raise ValueError("epsilon_decay must be in {}".format(self._parameter_constraints["epsilon_decay"]))
        
        if epsilon_decay_factor <= 0 or epsilon_decay_factor >= 1:
            raise Warning("epsilon_decay_factor should be in (0,1)")
        
        if epsilon <= 0 or epsilon >= 1:
            raise Warning("epsilon should be in (0,1)")
        
        if epsilon_min <= 0 or epsilon_min >= 1:
            raise Warning("epsilon_min should be in (0,1)")
        
        if w <= 0.5 or w >= 1:
            raise Warning("w should be in (0.5, 1)")
        
        if gamma <= 0 or gamma >= 1:
            raise Warning("gamma should be in (0,1)")
        

    # determine the next action
    def get_action(self, state):
        """
        Determine the next action based on the current state.

        Parameters
        ----------
        state : int or immutable object
            Current state.

        Returns
        -------
        action : int or immutable object
            Next action to be taken.
        """

        # action is from exploration
        # explore - do random action
        if np.random.rand() <= self.epsilon:
            
            action_index = np.random.choice(self.n_actions,1)[0]

            if self._USE_ACTION_DICT:
                action = self._action_dict.inverse[action_index]
            else:
                action = action_index 
        
            return action 

        # or action is from exploitation
        # exploit - choose action with max Q-value
        if self._USE_STATE_DICT:
            state_index = self._state_dict[state]
        else:
            state_index = state

        action_index = np.argmax(self.q_table[state_index]) 
        
        if self._USE_ACTION_DICT:
            action = self._action_dict.inverse[action_index]
        else:
            action = action_index


        return action

    # Q-Learning - update the Q Table using Q(s, a) and the learning rate alpha
    def update_q_table(self, state, action, reward, next_state = None) -> None:
        """
        Update the Q-table based on the observed state, action, reward, and next state.

        Parameters
        ----------
        state : int or immutable object
            Current state.

        action : int or immutable object
            Action taken in the current state.

        reward : float
            Reward received for taking the action in the current state.

        next_state : int or immutable object, default=None
            Next state after taking the action. If None, the previous state was a terminal state. 
        """
         
        # if dicts are used first convert state and action to indices
        if self._USE_STATE_DICT:
            state = self._state_dict[state]
        if self._USE_ACTION_DICT:
            action = self._action_dict[action]

        # Q(s, a) = reward + gamma * max_a' Q(s', a')
        if next_state == None:
            new_q_value = reward
        else:
            if np.amax(self.q_table[next_state]) == -np.inf:
                new_q_value = 0
            else:
                new_q_value = self.gamma * np.amax(self.q_table[next_state])

            new_q_value += reward
        
        last_q_value = self.q_table[state, action]
        if last_q_value == -np.inf:
            last_q_value = 0
    
        if self.alpha == 'polynomial':
            self.n_visited[state, action] += 1
            alpha = 1 / (np.power((self.n_visited[state, action]), self.w))
        else:
            alpha = self.alpha

        self.q_table[state, action] = (1 - alpha) * last_q_value + alpha * new_q_value

    # UI to dump Q Table contents
    def print_q_table(self) -> None:
        """
        Print the Q-table.
        """
        print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
        print(self.q_table)

    # update Exploration-Exploitation mix
    def update_epsilon(self) -> None:
        """
        Update the exploration-exploitation trade-off parameter (epsilon).
        """
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_factor
        else:
            self.epsilon = self.epsilon_min


