from collections import deque
from functions.utils import list_getter
import tensorflow as tf
import numpy as np
from games.adventurer_the_vegetarian.game import AdventurerTheVegetarian


class CNN:
    def build_cnn(self, num_actions):
        net = tf.layers.conv2d(self.state, 16, 5, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 32, 3, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 32, 3, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, 64, 3, 2, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)
        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, 128)
        self.output = tf.layers.dense(net, num_actions)


class Queue:
    def __init__(self, queue_capacity=10000):
        self.buffer = deque(maxlen=queue_capacity)
        self.count = 0

    def add(self, experience):
        self.buffer.append(experience)
        self.count += 1

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class VegetarianDeepQ(CNN, AdventurerTheVegetarian):
    def __init__(self, config):
        self.state_size = 65
        if config.phase == 'manual_play':
            self.manual_play()
        self.state_stack_num = 4
        self.deep_q()
        if config.phase == 'train':
            self.train_deep_q()
        elif config.phase == 'test':
            self.test_deep_q()
        else:
            raise ValueError('Unexpected phase')

    def config_train(self):
        self.train_episodes = 50000
        self.max_step_per_episode = 2000
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.00001
        self.batch_size = 64
        self.gamma = 0.99  # future reward discount
        self.lr = 0.00001  # learning rate
        self.save_dir = './checkpoints/vegetarian/Deep-Q'

    def config_test(self):
        self.max_step_per_episode = 2000
        self.num_repeat_episode = 10
        self.save_dir = './checkpoints/vegetarian/Deep-Q'

    def fill_initial_states(self):
        # raw_screen: capture of the game screen
        # frame: a single channel image converted from raw_screen
        # state_queue: a queue to enqueue and dequeue
        # current_state & next_state: h x w x state_stack_num

        while self.state_population.count < self.batch_size:
            self.initialize_game()
            state_queue = deque(maxlen=self.state_stack_num)
            # iterate to fill get very first state
            for i in range(self.state_stack_num):
                frame, _, _ = self.take_action(0)
                state_queue.append(frame)
            current_state = np.stack(state_queue, axis=2)

            for step in range(1, self.max_step_per_episode):
                random_action = np.random.randint(0, self.num_actions)
                next_frame, is_end, reward = self.take_action(random_action)  # this is next_state
                if is_end:
                    next_frame = np.zeros([self.state_size, self.state_size])
                    state_queue.append(next_frame)

                    next_state = np.stack(state_queue, axis=2)

                    self.state_population.add((current_state,
                                               random_action,
                                               reward,
                                               next_state))
                    break
                state_queue.append(next_frame)
                next_state = np.stack(state_queue, axis=2)
                self.state_population.add((current_state,
                                           random_action,
                                           reward,
                                           next_state))
                current_state = next_state

        print('initial Queue is filled')

    def deep_q(self, num_actions=3):
        self.state = tf.placeholder(tf.float32, [None, self.state_size, self.state_size, self.state_stack_num])
        self.actions = tf.placeholder(tf.int32, [None])
        self.q_hat = tf.placeholder(tf.float32, [None])
        self.is_train = tf.placeholder(tf.bool, None)
        self.num_actions = num_actions

    def train_deep_q(self):
        self.config_train()
        self.build_cnn(self.num_actions)
        self.state_population = Queue()
        self.fill_initial_states()

        onehot_actions = tf.one_hot(self.actions, self.num_actions)
        q_value = tf.reduce_sum(tf.multiply(self.output, onehot_actions), axis=1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.loss = tf.reduce_mean(tf.square(self.q_hat - q_value))
        optm = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.optm = tf.group([optm, update_ops])
        saver = tf.train.Saver(max_to_keep=1000)
        total_reward_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('train/reward_gain', total_reward_ph)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.save_dir)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        length = 0
        best_score = 0
        for episode in range(1, self.train_episodes):
            total_reward = 0.0

            self.initialize_game()
            # Note:
            # current_state[:, :, 1] = next_state[:, :, 0]
            # current_state[:, :, 2] = next_state[:, :, 1]
            # current_state[:, :, 3] = next_state[:, :, 2]
            state_queue = deque(maxlen=self.state_stack_num)
            # iterate to fill get very first state
            for i in range(self.state_stack_num):
                frame, _, _ = self.take_action(0)
                state_queue.append(frame)
            current_state = np.stack(state_queue, axis=2)
            for step in range(self.max_step_per_episode):
                length += 1
                # Explore or Exploit
                explore_prob = self.explore_stop + \
                               (self.explore_start - self.explore_stop) * \
                               np.exp(-self.decay_rate * length)
                if explore_prob > np.random.rand():
                    # explore and get random action
                    action = np.random.randint(self.num_actions)
                else:
                    # Get action from the model
                    feed = {self.state: np.expand_dims(current_state, axis=0),
                            self.is_train: False}
                    Qs = sess.run(self.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take an action, get a new state and the corresponding reward
                next_frame, is_end, reward = self.take_action(action)
                if is_end:
                    next_frame = np.zeros([self.state_size, self.state_size])
                    state_queue.append(next_frame)

                    next_state = np.stack(state_queue, axis=2)

                    self.state_population.add((current_state,
                                               action,
                                               reward,
                                               next_state))
                    if best_score < total_reward:
                        best_score = total_reward
                        saver.save(sess, self.save_dir + '/model', episode)
                    break
                state_queue.append(next_frame)

                next_state = np.stack(state_queue, axis=2)
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
                # episode_ends = (next_state_batch == np.zeros(current_state_batch[0].shape)).all(axis=(1, 2, 3))
                # target_Qs[episode_ends] = (0, 0, 0)
                end_game_index = rewards_batch < 0
                target_Qs[end_game_index] = (0, 0, 0)

                q_hat = rewards_batch + self.gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([self.loss, self.optm], feed_dict={self.state: current_state_batch,
                                                                      self.q_hat: q_hat,
                                                                      self.actions: actions_batch,
                                                                      self.is_train: True})

            print('Episode: {},'.format(episode),
                  'total_reward: {:.4f},'.format(total_reward),
                  'explor prob: {:.4f}'.format(explore_prob),
                  'meat add prob: {:.4f}'.format(self.meat_add_prob))
            summary_writer.add_summary(sess.run(summary_op, {total_reward_ph: total_reward}), episode)
            if episode % 100 == 0:
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
            ckpt = ckpt.replace('.index', '')
            restorer.restore(sess, ckpt.replace('.index', ''))
            total_reward = 0
            for episode in range(self.num_repeat_episode):
                self.initialize_game()
                state_queue = deque(maxlen=self.state_stack_num)
                # iterate to fill get very first state
                for i in range(self.state_stack_num):
                    frame, _, _ = self.take_action(0)
                    state_queue.append(frame)
                for step in range(self.max_step_per_episode):
                    state = np.stack(state_queue, axis=2)
                    action = np.argmax(sess.run(self.output, {self.state: np.expand_dims(state, axis=0), self.is_train: False}))
                    if is_end:
                        break
                    frame, is_end, reward = self.take_action(action)
                    state_queue.append(frame)
                    total_reward += reward
            ckpt_num = int(ckpt.split('-')[-1])
            print(ckpt_num)
            average_total_reward = float(total_reward) / float(self.num_repeat_episode)
            summary_writer.add_summary(sess.run(summary_op, {total_rewards_ph: average_total_reward}), ckpt_num)
        sess.close()

    def test_deep_q(self):
        self.config_test()
        self.build_cnn(self.num_actions)
        restorer = tf.train.Saver()
        total_rewards_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('test/total_reward', total_rewards_ph)
        sess = tf.Session()
        ckpt = list_getter(self.save_dir, 'index', 'model')[-1]
        ckpt = ckpt.replace('.index', '')
        restorer.restore(sess, ckpt.replace('.index', ''))
        total_reward = 0
        for episode in range(self.num_repeat_episode):
            self.initialize_game()
            state_queue = deque(maxlen=self.state_stack_num)
            # iterate to fill get very first state
            for i in range(self.state_stack_num):
                frame, _, _ = self.take_action(0)
                state_queue.append(frame)
            for step in range(self.max_step_per_episode):
                state = np.stack(state_queue, axis=2)
                action = np.argmax(sess.run(self.output, {self.state: np.expand_dims(state, axis=0), self.is_train: False}))
                frame, is_end, reward = self.take_action(action)
                # if is_end:
                #     break
                state_queue.append(frame)
                total_reward += reward
        sess.close()
        average_total_reward = float(total_reward) / float(self.num_repeat_episode)
        print("average total reward: %d" % average_total_reward)
