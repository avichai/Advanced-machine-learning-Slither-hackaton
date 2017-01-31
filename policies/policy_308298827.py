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
N_ACTIONS = 3
PATCH_RIB = 3
STATE_DIM = PATCH_RIB ** 2 + 1
BATCH_SIZE = 50
EPSILON_GREEDY = 0.1    # todo maybe epsilon reduces with time
DIRECTION_TO_IND = {direction:ind for ind,direction in enumerate(bp.Policy.TURNS)}


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


def get_nn_patch(state, head_pos):
    state_half_rib = PATCH_RIB // 2
    off = np.arange(PATCH_RIB) - state_half_rib
    r_offs = np.repeat(off, PATCH_RIB)
    c_offs = np.tile(off, PATCH_RIB)
    offs = np.vstack((r_offs, c_offs)).T
    offs += head_pos
    offs %= state.shape
    return state[offs[:, 0], offs[:, 1]]

    
class policy_308298827(bp.Policy):
    
    def cast_string_args(self, policy_args):
        # todo impl where from
        return policy_args


    def init_run(self):
        # todo init tf
        if PRINT: print('### INIT ###')
        self.state_dim = STATE_DIM
        self.session = tf.Session()
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='state') # todo consider int32
        self.scores = build_nn(self.states)              # todo rm SCREEN_WIDTH
        self.probabilities = tf.nn.softmax(self.scores, name='probabilities')
        self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
        self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.scores, self.taken_actions)       # todo check dims (scores is 3 while actions is 1)
        self.pg_loss = tf.reduce_sum(self.cross_entropy_loss * self.rewards)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.99, epsilon=1e-5)
        self.train_op = self.optimizer.minimize(self.pg_loss)
        self.saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())     # todo changed the initializer

        self.batch_ind = 0
        self.prev_t = -1
        self.prev_state = None
        self.prev_action = None
        self.batch_states = np.zeros((BATCH_SIZE, STATE_DIM), dtype=np.float32)
        self.batch_actions = np.zeros(BATCH_SIZE, dtype=np.int32)
        self.batch_rewards = np.zeros(BATCH_SIZE, dtype=np.float32)

        if PRINT: print('### INIT done###')

    def learn(self, reward, t):
        if PRINT: print('### LEARN ###')
        # todo check for time issues
        if self.batch_ind == BATCH_SIZE:
            self.batch_ind = 0
            self.session.run(self.train_op, {self.states: self.batch_states,
                                             self.taken_actions: self.batch_actions,
                                             self.rewards: self.batch_rewards})

        elif t - self.prev_t == 1:
            ind = self.batch_ind
            self.batch_states[ind] = self.prev_state
            self.batch_actions[ind] = self.prev_action
            self.batch_rewards[ind] = reward
            self.batch_ind += 1
        if PRINT: print('### LEARN done ###')


    def act(self, t, state, player_state):
        if PRINT: print('### ACT ###')
        head_pos = player_state['chain'][-1]
        direction = player_state['dir']           # todo use these features

        nn_state = np.zeros(STATE_DIM)
        nn_state[:-1] = get_nn_patch(state, head_pos)
        nn_state[-1] = DIRECTION_TO_IND[direction]
        nn_state = nn_state[np.newaxis]
        rand = np.random.rand()
        if rand < EPSILON_GREEDY:
            act_ind = 0 if rand < EPSILON_GREEDY / 3 else 1 if rand < 2 * EPSILON_GREEDY / 3 else 2
        else:
            probs = self.session.run(self.probabilities, feed_dict={self.states: nn_state})
            if PRINT: print(probs)
            act_ind = np.argmax(probs)

        self.prev_t = t
        self.prev_state = nn_state
        self.prev_action = act_ind
        if PRINT: print('### ACT done###')
        return bp.Policy.ACTIONS[act_ind]   # todo


    def get_state(self):
        # todo impl
        return self.session
