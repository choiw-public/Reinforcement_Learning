from collections import deque
import tensorflow as tf
import numpy as np
import random
from games.bubble_shooter.game import BubbleShooter
from functions.utils import list_getter
import time
import matplotlib.pyplot as plt


class CNN:
    def build_cnn(self, num_actions):
        net = tf.layers.conv2d(self.state, 32, 5, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 64, 3, 1, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 64, 5, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 128, 3, 1, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 128, 2, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 256)
        self.output = tf.layers.dense(net, num_actions)


class Queue:
    def __init__(self, queue_capacity=10000):
        self.buffer = deque(maxlen=queue_capacity)
        self.count = 0
        self.queue_capacity = queue_capacity

    def add(self, experience):
        self.buffer.append(experience)
        self.count += 1

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class BubbleDeepQ(CNN):
    def __init__(self, config):
        self.state_size = 37
        game = BubbleShooter()
        self.initialize_game = game.initialize_game
        self.manual_play = game.manual_play
        self.take_action = game.take_action
        if config.phase == 'manual_play':
            self.manual_play()
        if config.RL_type == 'policy_gradient':
            self.deep_q(num_actions=36)  # 0 to 140 by 4 interval
            if config.phase == 'train':
                self.train_deep_q()
            elif config.phase == 'test':
                self.test_deep_q()
            else:
                raise ValueError('Unexpected phase')
        else:
            raise NotImplemented()

    def config_train(self):
        self.train_episodes = 50000
        self.max_step_per_episode = 200
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.0001
        self.batch_size = 128
        self.gamma = 0.99  # future reward discount
        self.lr = 0.00003  # learning rate
        self.save_dir = './checkpoints/bubble/Deep-Q'

    def config_test(self):
        self.max_step_per_episode = 200
        self.num_repeat_episode = 5
        self.save_dir = './checkpoints/bubble/Deep-Q'

    def fill_state_population(self, sess=None, with_trained_model=False):
        # raw_screen: capture of the game screen
        # frame: a single channel image converted from raw_screen
        # state_queue: a queue to enqueue and dequeue
        # current_state & next_state: h x w x 3
        current_state = self.initialize_game()
        is_end = False
        while True:
            if not with_trained_model:
                action = random.randint(0, 35)
                angle = action * 4 + 20
            else:
                feed = {self.state: np.expand_dims(current_state, axis=0),
                        self.is_train: False}
                Qs = sess.run(self.output, feed_dict=feed)
                action = np.argmax(Qs)
                angle = action * 4 + 20
            next_state, is_end, reward = self.take_action(angle)  # this is next_state
            if is_end:
                self.state_population.add((current_state,
                                           action,
                                           reward,
                                           next_state))
                current_state = self.initialize_game()
            self.state_population.add((current_state,
                                       action,
                                       reward,
                                       next_state))
            current_state = next_state
            if self.state_population.count >= self.state_population.queue_capacity * 0.5:
                break
        print('initial Queue is filled')

    def deep_q(self, num_actions):
        self.state = tf.placeholder(tf.float32, [None, self.state_size, self.state_size, 6])
        self.actions = tf.placeholder(tf.int32, [None])  # actual action num not one-hot
        self.q_hat = tf.placeholder(tf.float32, [None])
        self.is_train = tf.placeholder(tf.bool, None)
        self.num_actions = num_actions

    def train_deep_q(self):
        self.config_train()
        self.build_cnn(self.num_actions)
        self.state_population = Queue()
        onehot_actions = tf.one_hot(self.actions, self.num_actions)
        q_value = tf.reduce_sum(tf.multiply(self.output, onehot_actions), axis=1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.loss = tf.reduce_mean(tf.square(self.q_hat - q_value))
        optm = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.optm = tf.group([optm, update_ops])
        saver = tf.train.Saver(max_to_keep=1000)
        total_reward_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('train/reward_gain', total_reward_ph)
        action_ph = tf.placeholder(tf.int64)
        tf.summary.histogram('train/action', action_ph)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.save_dir)

        sess = tf.Session()
        length = 0
        ckpt_list = list_getter(self.save_dir, 'index', 'model')
        if not ckpt_list:
            sess.run(tf.global_variables_initializer())
            self.fill_state_population()
            ep_start = 0
        else:
            length = 999999
            saver.restore(sess, ckpt_list[-1].replace('.index', ''))
            self.fill_state_population(sess, True)
            ep_start = int(ckpt_list[-1].split('-')[-1].split('.')[0])
        for episode in range(ep_start + 1, self.train_episodes):
            tic = time.time()
            action_record = []
            total_reward = 0.0
            current_state = self.initialize_game()
            # Note:
            # current_state[:, :, 1] = next_state[:, :, 0]
            # current_state[:, :, 2] = next_state[:, :, 1]
            # current_state[:, :, 3] = next_state[:, :, 2]
            for step in range(self.max_step_per_episode):
                length += 1
                # Explore or Exploit
                explore_prob = self.explore_stop + \
                               (self.explore_start - self.explore_stop) * \
                               np.exp(-self.decay_rate * length)
                if explore_prob > np.random.rand():
                    # explore and get random action
                    action = random.randint(0, 35)  # min and max shooting angle
                else:
                    # Get action from the model
                    feed = {self.state: np.expand_dims(current_state, axis=0),
                            self.is_train: False}
                    Qs = sess.run(self.output, feed_dict=feed)
                    action = np.argmax(Qs)
                    action_record.append(action)

                angle = action * 4 + 20  # angle changes by 4, minimum of 20 maxinum of 160
                next_state, is_end, reward = self.take_action(angle)

                if is_end:
                    self.state_population.add((current_state,
                                               action,
                                               reward,
                                               next_state))
                    break
                total_reward += reward
                self.state_population.add((current_state,
                                           action,
                                           reward,
                                           next_state))

                current_state = next_state

                # Sample mini-batch from state_queue
                batch = self.state_population.sample(self.batch_size)
                current_state_batch = np.array([each[0] for each in batch])
                actions_batch = np.array([each[1] for each in batch])
                rewards_batch = np.array([each[2] for each in batch])
                next_state_batch = np.array([each[3] for each in batch])

                # Q values for the next_state, which is going to be our target Q
                target_Qs = sess.run(self.output, feed_dict={self.state: next_state_batch,
                                                             self.is_train: True})
                end_game_index = rewards_batch < 0
                target_Qs[end_game_index] = np.zeros(self.num_actions)

                q_hat = rewards_batch + self.gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([self.loss, self.optm], feed_dict={self.state: current_state_batch,
                                                                      self.q_hat: q_hat,
                                                                      self.actions: actions_batch,
                                                                      self.is_train: True})

            print('Episode: {},'.format(episode),
                  'total_reward: {:.4f},'.format(total_reward),
                  'explor prob: {:.4f}'.format(explore_prob),
                  'duration: {:.4f}'.format(time.time() - tic))
            summary_writer.add_summary(sess.run(summary_op, {total_reward_ph: total_reward,
                                                             action_ph: action_record}), episode)
            if episode % 50 == 0:
                saver.save(sess, self.save_dir + '/model', episode)

    def test_deep_q_all(self):
        self.config_test()
        self.build_cnn(self.num_actions)
        restorer = tf.train.Saver()
        total_rewards_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('test/total_reward', total_rewards_ph)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.save_dir)
        sess = tf.Session()
        ckpt_list = list_getter(self.save_dir, 'index', 'model')
        for ckpt in ckpt_list:
            restorer.restore(sess, ckpt.replace('.index', ''))
            total_reward = 0
            for episode in range(self.num_repeat_episode):
                state = self.initialize_game()
                for step in range(self.max_step_per_episode):
                    feed = {self.state: np.expand_dims(state, axis=0),
                            self.is_train: False}
                    Qs = sess.run(self.output, feed_dict=feed)
                    action = np.argmax(Qs)
                    angle = action * 4 + 20
                    state, is_end, reward = self.take_action(angle)
                    if is_end:
                        break
            ckpt_num = int(ckpt.split('-')[-1].split('.')[0])
            print(ckpt_num)
            average_total_reward = float(total_reward) / float(self.num_repeat_episode)
            summary_writer.add_summary(sess.run(summary_op, {total_rewards_ph: average_total_reward}), ckpt_num)
        sess.close()

    def test_deep_q(self):
        self.config_test()
        self.build_cnn(self.num_actions)
        restorer = tf.train.Saver()
        sess = tf.Session()
        ckpt = list_getter(self.save_dir, 'index', 'model')[-1]
        restorer.restore(sess, ckpt.replace('.index', ''))
        total_reward = 0
        for episode in range(self.num_repeat_episode):
            state = self.initialize_game()
            for step in range(self.max_step_per_episode):
                feed = {self.state: np.expand_dims(state, axis=0),
                        self.is_train: False}
                Qs = sess.run(self.output, feed_dict=feed)
                action = np.argmax(Qs)
                angle = action * 4 + 20
                state, is_end, reward = self.take_action(angle)
                total_reward += reward
                if is_end:
                    break
                total_reward += reward
        print("average_total_reward =%.4f" % (float(total_reward) / float(self.num_repeat_episode)))
        sess.close()


class PolicyGradient:
    def policy_gradient(self):
        raise NotImplemented()

    def train_policy_gradient(self):
        raise NotImplemented()

    def test_policy_gradient(self):
        raise NotImplemented()
