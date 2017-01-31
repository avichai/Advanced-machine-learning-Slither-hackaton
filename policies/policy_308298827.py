#TODO:
# 1. remvoe asserts
# 2. search for todos
# 3. check runtime

from math import sqrt
from policies import base_policy as bp
import numpy as np
import tensorflow as tf

PRINT = 1

# todo set best hyper-params
EMPTY_SQUARE = 0
N_ACTIONS = 3
RAY_LEN = 1
BATCH_SIZE = 50  # todo default 50
EPSILON_GREEDY_INIT = 1    # todo maybe epsilon reduces with time
DIRECTION_TO_IND = {direction:ind for ind,direction in enumerate(bp.Policy.TURNS)}
GAMMA = 0.9


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


def affine(name_scope, input_tensor, out_channels, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / sqrt(float(input_channels))), name='weights')
        biases = tf.Variable(tf.zeros([out_channels]), name='biases')

        output_tensor = tf.matmul(input_tensor, weights) + biases
        if relu:
            output_tensor = tf.nn.relu(output_tensor)
        return output_tensor


def build_nn(state):
    h1 = affine('part1', state, 500)
    h2 = affine('part2', h1, 100)
    s = affine('scores', h2, N_ACTIONS, relu=False)
    return s


def get_nn_state(state, head_pos, direction, dir_offsets):
    offs = dir_offsets[direction]
    head_offs = offs + head_pos
    head_offs %= state.shape
    return state[head_offs[:, 0], head_offs[:, 1]]


def manipulate_reward(r):
    for t in range(len(r)-2,-1,-1):
        r[t] += GAMMA * r[t+1]

    
class policy_308298827(bp.Policy):
    
    def cast_string_args(self, policy_args):
        policy_args['ray_len'] = int(policy_args['ray_len']) if 'ray_len' in policy_args else RAY_LEN
        return policy_args


    def init_run(self):
        # todo init tf
        #if PRINT: print('### INIT ###')
        self.state_dim = self.ray_len * N_ACTIONS + 1
        self.dir_offsets = get_dir_offsets(self.ray_len)
        
        self.session = tf.Session()
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='state') # todo consider int32
        self.scores = build_nn(self.states)              # todo rm SCREEN_WIDTH
        self.probabilities = tf.nn.softmax(self.scores, name='probabilities')
        self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
        self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.scores, self.taken_actions)       # todo check dims (scores is 3 while actions is 1)
        self.pg_loss = tf.reduce_sum(self.cross_entropy_loss * self.rewards)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.99, epsilon=1e-5) # todo default lr = 0.0001
        self.train_op = self.optimizer.minimize(self.pg_loss)
        # self.saver = tf.train.Saver()   # todo rm

        self.session.run(tf.global_variables_initializer())     # todo changed the initializer

        self.batch_ind = 0
        self.prev_t = -1
        self.prev_state = None
        self.prev_action = None
        self.batch_states = np.zeros((BATCH_SIZE, self.state_dim), dtype=np.float32)
        self.batch_actions = np.zeros(BATCH_SIZE, dtype=np.int32)
        self.batch_rewards = np.zeros(BATCH_SIZE, dtype=np.float32)

        self.round = 0
        self.epsilon_greedy = EPSILON_GREEDY_INIT
        # self.explore = True
        #if PRINT: print('### INIT done###')

    def learn(self, reward, t):
        #if PRINT: print('### LEARN ###')
        # todo check for time issues
        if self.batch_ind == BATCH_SIZE:
            self.batch_ind = 0
            manipulate_reward(self.batch_rewards)
            self.session.run(self.train_op, {self.states: self.batch_states,
                                             self.taken_actions: self.batch_actions,
                                             self.rewards: self.batch_rewards})

        elif t - self.prev_t == 1:
            ind = self.batch_ind
            self.batch_states[ind] = self.prev_state
            self.batch_actions[ind] = self.prev_action
            self.batch_rewards[ind] = reward
            self.batch_ind += 1
        # if PRINT: print('REWARD:', reward)
        #if PRINT: print('### LEARN done ###')


    def act(self, t, state, player_state):
        #if PRINT: print('### ACT ###')
        self.round += 1     # todo: shouldnt increment forever (could explode)
        if self.round > 1000:
            # self.explore = False
            self.epsilon_greedy = .01

        head_pos = player_state['chain'][-1]
        direction = player_state['dir']           # todo use these features

        nn_state = np.zeros(self.state_dim)
        nn_state[:-1] = get_nn_state(state, head_pos, direction, self.dir_offsets)
        nn_state[-1] = state[head_pos % state.shape]
        nn_state = nn_state[np.newaxis]

        rand = np.random.rand()
        # if self.explore and (rand < EPSILON_GREEDY):
        if rand <= self.epsilon_greedy:
            mat_state = nn_state[0][:-1].reshape((N_ACTIONS, self.ray_len))
            objs = (mat_state != EMPTY_SQUARE)
            if not np.any(objs):
                act_ind = 0 if rand < self.epsilon_greedy / 3 else 1 if rand < 2 * self.epsilon_greedy / 3 else 2
            else:
                dir_min_dist = np.argmax(objs, axis=1)
                no_obj = np.logical_not(np.any(objs, axis=1))
                dir_min_dist[no_obj] = self.ray_len + 1
                act_ind = np.argmin(dir_min_dist)
                # print('DIR MIN DIST:', dir_min_dist)
                # print('ACT IND:', act_ind)
                # print('ACTION:', bp.Policy.ACTIONS[act_ind])
                # print('ME:', state[head_pos % state.shape])
        else:
            probs = self.session.run(self.probabilities, feed_dict={self.states: nn_state})
            if PRINT: print('PROBS:', probs)
            m = np.max(probs)
            if np.sum(probs == m) >= 2:
                a = 1
            act_ind = np.argmax(probs)

        self.prev_t = t
        self.prev_state = nn_state
        self.prev_action = act_ind
        #if PRINT: print('### ACT done###')
        # if PRINT: print('ROUND:' , self.round, 'EXPLORE:', self.explore)
        if PRINT: print('ROUND:' , self.round, 'EPSILON:', self.epsilon_greedy)
        # print('rl', self.ray_len)
        return bp.Policy.ACTIONS[act_ind]   # todo


    def get_state(self):
        return self
