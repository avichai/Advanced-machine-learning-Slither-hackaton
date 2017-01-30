#TODO:
# 1. remvoe asserts

from policies import base_policy as bp
import numpy as np


class policy_308298827(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        self.q = np.zeros((64, 3))   # todo maybe random
        self.state_t_1 = None
        self.state_t = None
        self.act_t_1 = None
        self.act_t = None
        self.items = [0, 100, 1, -1]
        self.state_dict = {}
        l = 0
        for i in self.items:
            for j in self.items:
                for k in self.items:
                    self.state_dict[(i, j, k)] = l
                    l += 1

    def learn(self, reward, t):
        if self.state_t is None or self.state_t_1 is None or self.act_t is None or self.act_t_1 is None:
            return

        delta = self.q[self.state_t, self.act_t] - reward - bp.GAMMA * np.max(self.q[self.state_t_1])
        self.q[self.state_t, self.act_t] -= bp.RATE * delta


    def act(self, t, state, player_state):

        chain, dir = player_state['chain'], player_state['dir']
        head_pos = chain[-1]
        state_shape = state.shape

        my_state = np.zeros(3)
        turn_dir = bp.Policy.TURNS[dir]
        my_state[0] = state[head_pos.move(turn_dir['CC']) % state_shape]
        my_state[1] = state[head_pos.move(turn_dir['CW']) % state_shape]
        my_state[2] = state[head_pos.move(turn_dir['CN']) % state_shape]

        my_state[(my_state>0) & (my_state!=100)] = 1
        my_state[my_state<0] = -1
        my_state = tuple(my_state)

        assert(my_state in self.state_dict)
        state_ind = self.state_dict[my_state]
        act_ind = np.argmax(self.q[state_ind])
        act = bp.Policy.ACTIONS[act_ind]

        self.state_t = self.state_t_1
        self.state_t_1 = state_ind

        self.act_t = self.act_t_1
        self.act_t_1 = act_ind

        return act


    def get_state(self):
        return self.q
