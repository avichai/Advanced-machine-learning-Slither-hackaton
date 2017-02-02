from policies import base_policy as bp
import numpy as np
from pickle import load


EMPTY_SQUARE = 0
N_ACTIONS = 3
N_ACTIONS_ARR = np.arange(N_ACTIONS)

BATCH_SIZE = 10
EPSILON_GREEDY_INIT = 1
EPSILON_GREEDY_REDUCE_FACTOR = .99999
EARLY_EXPLORATION_N_ROUNDS = 50000

RAY_LEN = 10
LEARNING_RATE = 0.01
GAMMA = 0.9
MANIPULATE_RESULTS = True


def get_q_action_init():
    return ((np.random.rand(3) - 0.5) * 0.1).astype(np.float32)


def get_dir_offsets(ray_len):
    ray_zeroes = np.zeros(ray_len, dtype=np.int32)
    ray_offs = np.arange(ray_len, dtype=np.int32) + 1

    r_offs = np.vstack((ray_zeroes, ray_offs)).T
    c_offs = np.vstack((ray_offs, ray_zeroes)).T

    n_offs = np.vstack((-r_offs, r_offs, -c_offs))
    e_offs = np.vstack((-c_offs, c_offs, r_offs))
    s_offs = np.vstack((r_offs, -r_offs, c_offs))
    w_offs = np.vstack((c_offs, -c_offs, -r_offs))

    dir_offsets = {
        'N': n_offs,
        'E': e_offs,
        'S': s_offs,
        'W': w_offs,
    }
    return dir_offsets


def get_q_state(state, head_pos, direction, dir_offsets, ray_len):
    offs = dir_offsets[direction]
    head_offs = offs + head_pos
    head_offs %= state.shape
    rays = state[head_offs[:, 0], head_offs[:, 1]]
    
    rays_mat = rays.reshape((N_ACTIONS, ray_len))
    objs = (rays_mat != EMPTY_SQUARE)
    min_dist = np.argmax(objs, axis=1)
    near_objs = rays_mat[N_ACTIONS_ARR, min_dist]
    is_realy_near = (min_dist == 0).astype(np.int32)

    q_state = np.hstack((near_objs, is_realy_near))
    return np_to_string(q_state), rays_mat


def manipulate_reward(reward, chain):
    if reward < 0:
        return reward

    l = 1
    lc = len(chain)
    for i in range(lc-1):
        ci = chain[i]
        ci1 = chain[i+1]
        l += abs(ci1[0] - ci[0]) + abs(ci1[1] - ci[1])
    return reward - l


def np_to_string(array):
    return ''.join([str(e) for e in array])


def get_explore_act_ind(policy, q_state, rand, state, head_pos):
    mat_state = q_state.reshape((N_ACTIONS, policy.ray_len))
    objs = (mat_state != EMPTY_SQUARE) & (mat_state != state[head_pos % state.shape])
    if not np.any(objs) or rand < 0.01:
        return 0 if rand < 1/3 else 1 if rand < 2/3 else 2
    else:
        dir_min_dist = np.argmax(objs, axis=1)
        no_obj = np.logical_not(np.any(objs, axis=1))
        dir_min_dist[no_obj] = policy.ray_len + 1
        return np.argmin(dir_min_dist)


class policy_308298827(bp.Policy):

    def cast_string_args(self, policy_args):
        self.ray_len = int(policy_args['rl']) if 'rl' in policy_args else RAY_LEN
        self.learning_rate = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        self.manipulate_reward = int(policy_args['mr']) if 'mr' in policy_args else MANIPULATE_RESULTS
        self.gamma = float(policy_args['gm']) if 'gm' in policy_args else GAMMA

        self.early_exploration_n_rounds = EARLY_EXPLORATION_N_ROUNDS
        self.epsilon_greedy = EPSILON_GREEDY_INIT
        self.Q = {}
        if 'load_from' in policy_args:
            self.early_exploration_n_rounds = 0
            self.epsilon_greedy = 0.01
            try:
                self.Q = load(open(policy_args['load_from'], 'rb'))
            except:
                pass

        return policy_args


    def init_run(self):
        self.state_dim = self.ray_len * N_ACTIONS
        self.dir_offsets = get_dir_offsets(self.ray_len)
        self.early_round = 0

        self.ts_learn = -10
        self.ts_act = -10
        self.state_t = None
        self.state_t1 = None
        self.action_ind_t = None
        self.reward_t = None
        
        self.batch_ind = 0
        self.batch_states = []
        self.batch_action_inds = []
        self.batch_updates = []


    def learn(self, reward, t):
        if self.batch_ind == BATCH_SIZE:
            self.batch_ind = 0

            for i in range(BATCH_SIZE):
                state_probs = self.Q[self.batch_states[i]]
                state_probs[self.batch_action_inds[i]] += self.batch_updates[i]

            self.batch_states = []
            self.batch_action_inds = []
            self.batch_updates = []
            
        self.reward_t = reward
        self.ts_learn = t


    def act(self, t, state, player_state):
        chain = player_state['chain']
        head_pos = chain[-1]
        direction = player_state['dir']

        q_state_str, rays_mat = get_q_state(state, head_pos, direction, self.dir_offsets, self.ray_len)
        if q_state_str not in self.Q:
            self.Q[q_state_str] = get_q_action_init()

        rand = np.random.rand()
        if self.early_round < self.early_exploration_n_rounds:
            self.early_round += 1
            act_ind = get_explore_act_ind(self, rays_mat, rand, state, head_pos)
        elif rand < self.epsilon_greedy:
            self.epsilon_greedy *= EPSILON_GREEDY_REDUCE_FACTOR
            act_ind = 0 if rand < self.epsilon_greedy / 3 else 1 if rand < 2 * self.epsilon_greedy / 3 else 2
        else:
            act_ind = np.argmax(self.Q[q_state_str])

        # sync validation
        if (t == self.ts_learn) and (t - self.ts_act == 1):
            self.batch_ind += 1

            m_reward = manipulate_reward(self.reward_t, chain) if self.manipulate_reward else self.reward_t
            delta = self.Q[self.state_t][self.action_ind_t] - m_reward - self.gamma * np.max(self.Q[q_state_str])
            self.batch_states.append(self.state_t)
            self.batch_action_inds.append(self.action_ind_t)
            self.batch_updates.append(- self.learning_rate * delta)

        self.ts_act = t
        self.state_t = q_state_str
        self.action_ind_t = act_ind

        return bp.Policy.ACTIONS[act_ind]


    def get_state(self):
        return self.Q

