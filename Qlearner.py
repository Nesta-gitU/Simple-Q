import numpy as np

class Qlearner():

    _parameter_constraints: dict = {
        "alpha": ['polynomial'], #or int, float 
        "epsilon_decay": ['power'] 
    }



    def __init__(self, 
                 states, #int or list of immutable objects
                 actions, #int or list of immutable objects
                 alpha = 'polynomial', 
                 w = 0.85, 
                 gamma = 0.8,
                 epsilon_decay = 'power',
                 epsilon = 0.9,
                 epsilon_decay_factor = 0.9,
                 epsilon_min = 0.1): 
    
        self.n_actions = actions
        self.n_states = states
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.epislon_decay_factor = epsilon_decay_factor
        self.epsilon_min = epsilon_min

        self.q_table = np.full([self.n_states, self.n_actions], fill_value= None)
        
        if alpha == 'polynomial':
            self.n_visited = np.zeros([self.n_states, self.n_actions])
            self.w = w

    # determine the next action
    def get_action(self, state: int) -> int:
        # action is from exploration
        if np.random.rand() <= self.epsilon:
            # explore - do random action
            return np.random.choice(self.n_actions,1)[0]

        # or action is from exploitation
        # exploit - choose action with max Q-value
        return np.argmax(self.q_table[state])

    # Q-Learning - update the Q Table using Q(s, a) and the learning rate alpha
    def update_q_table(self, state: int, action: int, reward: float, next_state = None) -> None:
        # Q(s, a) = reward + gamma * max_a' Q(s', a')
        if next_state == None:
            new_q_value = reward
        else:
            if np.amax(self.q_table[next_state]) == None:
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
    def print_q_table(self):
        print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
        print(self.q_table)

    # update Exploration-Exploitation mix
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


