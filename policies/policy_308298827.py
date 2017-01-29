from policies import base_policy as bp
import numpy as np


class policy_308298827(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        pass

    def learn(self, reward, t):
        print(reward)


    def act(self, t, state, player_state):
        print(state)
        print(player_state)

    def get_state(self):
        return None
