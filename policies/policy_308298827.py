#TODO:
# 1. remvoe asserts
# 2. search for todos
# 3. check runtime

from policies import base_policy as bp
import numpy as np
from pickle import load


PRINT = 1

EMPTY_SQUARE = 0
N_ACTIONS = 3
BATCH_SIZE = 50
EPSILON_GREEDY_INIT = 1
DIRECTION_TO_IND = {direction: ind for ind,direction in enumerate(bp.Policy.TURNS)}  # todo needed ?
GAMMA = 0.5
Q_ACTION_INIT = [0, 0, 0]

RAY_LEN = 3
LEARNING_RATE = 0.01


def get_dir_offsets(ray_len):
    ray_zeroes = np.zeros(ray_len, dtype=np.int32)
    ray_offs = np.arange(ray_len, dtype=np.int32) + 1

    r_offs = np.vstack((ray_zeroes, ray_offs)).T
    c_offs = np.vstack((ray_offs, ray_zeroes)).T

    dir_offsets = {
        'N': np.vstack((-r_offs, r_offs, -c_offs)),
        'E': np.vstack((-c_offs, c_offs, r_offs)),
        'S': np.vstack((r_offs, -r_offs, c_offs)),
        'W': np.vstack((c_offs, -c_offs, -r_offs)),
    }
    return dir_offsets


def get_q_state(state, head_pos, direction, dir_offsets):
    offs = dir_offsets[direction]
    head_offs = offs + head_pos
    head_offs %= state.shape
    return state[head_offs[:, 0], head_offs[:, 1]]


def manipulate_reward(r):
    for t in range(len(r)-2,-1,-1):
        r[t] += GAMMA * r[t+1]


class policy_308298827(bp.Policy):

    def cast_string_args(self, policy_args):
        self.ray_len = int(policy_args['ray_len']) if 'ray_len' in policy_args else RAY_LEN
        self.learning_rate = float(policy_args['learning_rate']) if 'learning_rate' in policy_args else LEARNING_RATE
        if 'load_from' in policy_args:
            try:
                self.Q = load(open(policy_args['load_from'], 'rb'))
            except IOError:
                self.Q = {}
        return policy_args


    def init_run(self):
        # todo init tf
        #if PRINT: print('### INIT ###')
        self.state_dim = self.ray_len * N_ACTIONS
        self.dir_offsets = get_dir_offsets(self.ray_len)

        self.ts_learn = -1
        self.st_act = -1
        self.state_t = None
        self.state_t1 = None
        self.action_ind_t = None
        self.reward_t = None
        
        self.batch_ind = 0
        self.states = []
        self.actions = []
        self.updates = []

        self.round = 0
        self.epsilon_greedy = EPSILON_GREEDY_INIT
        # self.explore = True
        #if PRINT: print('### INIT done###')

    def learn(self, reward, t):
        #if PRINT: print('### LEARN ###')
        # todo check for time issues
        if self.batch_ind == BATCH_SIZE:
            self.batch_ind = 0

            # todo

            self.states = []
            self.actions = []
            self.updates = []
            # manipulate_reward(self.batch_rewards) # todo check if needed
            
        self.reward_t = reward
        self.ts_learn = t
        #if PRINT: print('### LEARN done ###')


    def act(self, t, state, player_state):
        #if PRINT: print('### ACT ###')
        self.round += 1     # todo: shouldnt increment forever (could explode)

        head_pos = player_state['chain'][-1]
        direction = player_state['dir']           # todo use these features

        q_state = get_q_state(state, head_pos, direction, self.dir_offsets)

        rand = np.random.rand()
        # if rand <= self.epsilon_greedy:
        if self.round < 1000:
            mat_state = q_state.reshape((N_ACTIONS, self.ray_len))
            objs = (mat_state != EMPTY_SQUARE) #& (mat_state != state[head_pos % state.shape])
            if not np.any(objs):
                    act_ind = 0 if rand < self.epsilon_greedy / 3 else 1 if rand < 2 * self.epsilon_greedy / 3 else 2
            else:
                dir_min_dist = np.argmax(objs, axis=1)
                no_obj = np.logical_not(np.any(objs, axis=1))
                dir_min_dist[no_obj] = self.ray_len + 1
                act_ind = np.argmin(dir_min_dist)
                print('DIR MIN DIST:', dir_min_dist)
            print('HEAD POS:', head_pos % state.shape)
            print('DIR:', direction)
            print('ACTION:', bp.Policy.ACTIONS[act_ind])
        else:
            act_ind = np.argmax(self.Q[q_state])


        if (t == self.ts_reward) and (t - self.ts_act == 1):
            self.batch_ind += 1

            if self.state_t not in self.Q:
                self.Q[self.state_t] = Q_ACTION_INIT
            if q_state not in self.Q:
                self.Q[q_state] = Q_ACTION_INIT

            delta = self.Q[self.state_t][self.action_ind_t] - self.reward_t - GAMMA * np.max(self.q[q_state])
            self.q_states.append(self.state_t)
            self.q_action_inds.append(self.action_ind_t)
            self.q_updates.append(- LEARNING_RATE * delta)

        self.st_act = t
        self.state_t = q_state
        self.action_ind_t = act_ind

        #if PRINT: print('### ACT done###')
        if PRINT: print('ROUND:', self.round, 'EPSILON:', self.epsilon_greedy)
        return bp.Policy.ACTIONS[act_ind]


    def get_state(self):
        return self.Q
