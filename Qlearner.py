import numpy as np
from bidict import bidict

class Qlearner():

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
                 w = 0.85, 
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
        self.epislon_decay_factor: float = epsilon_decay_factor
        self.epsilon_min: float = epsilon_min

        self.q_table = np.full([self.n_states, self.n_actions], fill_value= -np.inf)
        
        if alpha == 'polynomial':
            self.n_visited = np.zeros([self.n_states, self.n_actions])
            self.w = w

    # determine the next action
    def get_action(self, state) -> int:
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
            action_index = np.argmax(self.q_table[state_index])
            action = self._action_dict.inverse[action_index]
        else:
            action_index = np.argmax(self.q_table[state]) 
            action = action_index

        return action

    # Q-Learning - update the Q Table using Q(s, a) and the learning rate alpha
    def update_q_table(self, state, action, reward, next_state = None) -> None:
        # if dicts are used first convert state and action to indices
        if self._USE_STATE_DICT:
            state = self._state_dict[state]
        if self._USE_ACTION_DICT:
            action = self._action_dict[action]

        # Q(s, a) = reward + gamma * max_a' Q(s', a')
        if next_state == None:
            new_q_value = reward
        else:
            if np.amax(self.q_table[next_state]) == np.inf:
                new_q_value = 0
            else:
                new_q_value = self.gamma * np.amax(self.q_table[next_state])

            new_q_value += reward
    
        if self.alpha == 'polynomial':
            self.n_visited[state, action] += 1
            alpha = 1 / (np.power((self.n_visited[state, action]), self.w))
        else:
            alpha = self.alpha

        self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * new_q_value

    # UI to dump Q Table contents
    def print_q_table(self) -> None:
        print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
        print(self.q_table)

    # update Exploration-Exploitation mix
    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


