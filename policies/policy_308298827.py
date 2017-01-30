#TODO:
# 1. remvoe asserts

from policies import base_policy as bp
import numpy as np
import tensorflow as tf


class policy_308298827(bp.Policy):

    def cast_string_args(self, policy_args):
        return policy_args

    def init_run(self):
        self.q = np.random.rand(192).reshape((64, 3))   # todo maybe random
        
        self.t = None
        self.state_t = None
        self.state_t_1 = None
        self.act_t = None
        self.act_t_1 = None

        self.batch_size = 10
        self.q_r_inds = []
        self.q_c_inds = []
        self.q_updates = []
        
        self.items = [0, 100, 1, -1]
        self.state_dict = {}
        l = 0
        for i in self.items:
            for j in self.items:
                for k in self.items:
                    self.state_dict[(i, j, k)] = l
                    l += 1

    def learn(self, reward, t):
        if self.state_t is None or self.state_t_1 is None:
            return
        if self.act_t is None or self.act_t_1 is None:
            return
        if self.t is None:
            return
        if t - self.t !=1:
            return

        # print('inds len')
        # print(len(self.q_r_inds))
        # print(len(self.q_c_inds))
        # print(self.q_updates)
        assert(len(self.q_r_inds) == len(self.q_c_inds))
        assert(len(self.q_r_inds) == len(self.q_updates))

        if len(self.q_r_inds) == self.batch_size:
            print('UPDATING')
            self.q[self.q_r_inds, self.q_c_inds] += self.q_updates
            self.q_r_inds = []
            self.q_c_inds = []
            self.q_updates = []
        else:
            print('COLLECTING')
            delta = self.q[self.state_t, self.act_t] - reward - bp.GAMMA * np.max(self.q[self.state_t_1])
            self.q_r_inds.append(self.state_t)
            self.q_c_inds.append(self.act_t)
            self.q_updates.append(- bp.RATE * delta)

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

        if np.random.rand() < bp.EPSILON:
            act_ind = np.random.randint(0,3)
        else:
            act_ind = np.argmax(self.q[state_ind])
        act = bp.Policy.ACTIONS[act_ind]

        self.state_t = self.state_t_1
        self.state_t_1 = state_ind
        
        self.act_t = self.act_t_1
        self.act_t_1 = act_ind

        self.t = t

        return act

    def get_state(self):
        return self.q


##########################################################
SAVE_PATH = "/tmp/Pong_PolicyGradient/"


def define_policy_network(state):
    num_actions = NUM_ACTIONS
    h1 = Affine('part1', state, 500)
    h2 = Affine('part2', h1, 100)
    s = Affine('scores', h2, num_actions, relu=False)
    return s


class Policy:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.session = tf.Session()
        self.state = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='state')
        self.scores = define_policy_network(self.state)         # todo maybe normalize
        self.probabilities = tf.nn.softmax(self.scores, name='probabilities')
        self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
        self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.scores, self.taken_actions)
        self.pg_loss = tf.reduce_sum(self.cross_entropy_loss * self.rewards)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.99, epsilon=1e-5)
        self.train_op = self.optimizer.minimize(self.pg_loss)
        self.saver = tf.train.Saver()

    def init_weights(self, snapshot_file=""):
        if snapshot_file != "":
            self.saver.restore(self.session, snapshot_file)
        else:
            self.session.run(tf.initialize_all_variables())

    def predict(self, states_feed):
        p = self.session.run(self.probabilities, feed_dict={self.state: states_feed})
        return np.argmax(p, axis=1), p

    def run_episode(self, render=False):
        self._sim.reset()
        MAX_STEPS = 300
        episode_states = np.zeros((MAX_STEPS, self.state_dim), dtype=np.float32)
        episode_actions = np.zeros((MAX_STEPS,), dtype=np.float32)
        episode_rewards = np.zeros((MAX_STEPS,), dtype=np.float32)
        for t in range(MAX_STEPS):
            if render:
                self._sim.render(0.01, title="Score: {}/{}".format(self._score[0], self._score[1]))
            if self._sim.done():
                self._score[0] += self._sim.left_win()
                self._score[1] += self._sim.right_win()
                if render:
                    self._sim.render(1.0, title="Score: {}/{}".format(self._score[0], self._score[1]))
                t = t - 1
                break
            state = self._sim.get_state()
            episode_states[t] = state
            action_left = self._policy_left.get_action(state)
            _, prob_right = self.predict(state.reshape(1, self.state_dim))
            action_right = np.argmax(np.random.multinomial(1, prob_right[0] - 1e-5))
            episode_actions[t] = action_right
            self._sim.step(action_left, action_right)
            ball_close_to_paddle = abs(self._sim._ball._y - self._sim._right_paddle._y) <= 0.5 * PADDLE_HEIGHT
            episode_rewards[t] = (self._sim._ball._x >= SCREEN_WIDTH - 1) * (2.0 * ball_close_to_paddle - 1.0)
        return episode_states[:t + 1], episode_actions[:t + 1], episode_rewards[:t + 1]

    def manipulate_reward(self, r):
        gamma = 0.9
        for t in range(len(r) - 2, -1, -1):
            r[t] += gamma * r[t + 1]

    def policy_gradient(self, batchsize):
        episode_states = np.zeros((0, self.state_dim), dtype=np.float32)
        episode_actions = np.zeros((0,), dtype=np.float32)
        episode_rewards = np.zeros((0,), dtype=np.float32)
        for _ in range(batchsize):
            s, a, r = self.run_episode()
            self.manipulate_reward(r)
            episode_states = np.vstack((episode_states, s))
            episode_actions = np.hstack((episode_actions, a))
            episode_rewards = np.hstack((episode_rewards, r))
        episode_rewards *= 1.0 / batchsize
        self.session.run(self.train_op, {self.state: episode_states,
                                         self.taken_actions: episode_actions,
                                         self.rewards: episode_rewards})


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_file", default='', help='Restore snapshot')
    parser.add_argument("--batchsize", type=int, default=10, help='Number of episodes for estimating the gradient')
    parser.add_argument("--num_iters", type=int, default=10000, help='How many SGD iterations to perform')
    parser.add_argument("--play_interval", type=int, default=10, help='Frequency of playing an episode')
    parser.add_argument("--save_frequency", type=int, default=100, help='Frequency of saving a snapshot')
    parser.add_argument("--play_only", action='store_true', help='Only play the learned policy')
    args = parser.parse_args()
    return args


def main(args):
    PG = PolicyGradient()
    PG.init_weights(snapshot_file=args.snapshot_file)
    if args.play_only:
        for t in range(args.num_iters):
            PG.run_episode(render=True)
    else:
        for t in range(args.num_iters):
            PG.policy_gradient(args.batchsize)
            if t % args.play_interval == 0:
                PG.run_episode(render=True)
            if t > 0 and t % args.save_frequency == 0:
                PG.saver.save(PG.session, SAVE_PATH + "/model_{}.ckpt".format(t))
