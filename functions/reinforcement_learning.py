from collections import deque
import tensorflow as tf
import numpy as np
import time


class CNN:
    def build_cnn(self, num_actions):
        root = tf.layers.conv2d(self.states, 16, 3, 2, 'same')
        net = tf.layers.batch_normalization(root, training=self.is_train)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 16, 3, 2, 'same')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 32, 3, 2, 'same')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 64, 3, 2, 'same')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(net, 128, 5, 1, 'valid')
        net = tf.layers.batch_normalization(net, training=self.is_train)
        net = tf.nn.relu(net)
        self.output = tf.reshape(tf.layers.conv2d(net, num_actions, 1, 1, 'valid'), [-1, num_actions])


class DeepQ:
    def config_train(self):
        self.train_episodes = 50000
        self.max_step_per_episode = 300
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.0001
        self.batch_size = 64
        self.gamma = 0.99  # future reward discount
        self.save_dir = './checkpoints/Deep-Q'

    def config_test(self):
        self.max_step_per_episode = 500
        self.num_repeat_episode = 5
        self.save_dir = './checkpoints/Deep-Q'

    def deep_q(self, lr=0.0006, num_actions=3):
        self.states = tf.placeholder(tf.float32, [None, *self.state_shape, self.stack_frame_num])
        self.actions = tf.placeholder(tf.int32, [None])
        self.q_hat = tf.placeholder(tf.float32, [None])
        self.is_train = tf.placeholder(tf.bool, None)
        self.num_actions = num_actions
        onehot_actions = tf.one_hot(self.actions, num_actions)
        self.build_cnn(num_actions)

        q_value = tf.reduce_sum(tf.multiply(self.output, onehot_actions), axis=1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.loss = tf.reduce_mean(tf.square(self.q_hat - q_value))
        optm = tf.train.RMSPropOptimizer(lr).minimize(self.loss)
        self.optm = tf.group([optm, update_ops])

    def train_deep_q(self):
        self.config_train()
        saver = tf.train.Saver()
        total_rewards_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('train/total_rewards', total_rewards_ph)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.save_dir)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.initialize_game()
        raw_screen = self.take_action(np.random.randint(0, self.num_actions), 0)
        frame = self.get_frame(raw_screen, self.state_shape)

        state_queue = deque([frame for _ in range(self.stack_frame_num)], maxlen=self.stack_frame_num)
        length = 0
        for episode in range(1, self.train_episodes):
            tic = time.time()
            total_reward = 0
            step = 1
            while step < self.max_step_per_episode:
                length += 1
                # Explore or Exploit
                explore_prob = self.explore_stop + \
                               (self.explore_start - self.explore_stop) * \
                               np.exp(-self.decay_rate * length)
                if explore_prob > np.random.rand():
                    # explore and get random action
                    action = np.random.randint(self.num_actions)
                else:
                    # Get action from Q-network
                    feed = {self.states: np.expand_dims(current_state, axis=0), self.is_train: False}
                    Qs = sess.run(self.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                raw_screen = self.take_action(action, step)
                if self.is_collision:
                    reward = -50
                    self.is_collision = False
                else:
                    reward = 1
                total_reward += reward
                current_state = np.stack(state_queue, axis=2)
                next_frame = self.get_frame(raw_screen, self.state_shape)
                state_queue.append(next_frame)
                next_state = np.stack(state_queue, axis=2)
                if self.is_collision:
                    # Add experience to memory
                    self.state_queue.add((current_state, action, reward, next_state))

                else:
                    # Add experience to memory
                    if step > 30:  # initial frames are not useful so after 30 frames are only used
                        self.state_queue.add((current_state, action, reward, next_state))
                    current_state = next_state
                    # step += 1

                # Sample mini-batch from memory
                batch = self.state_queue.sample(self.batch_size)
                current_state_batch = np.array([each[0] for each in batch])
                actions_batch = np.array([each[1] for each in batch])
                rewards_batch = np.array([each[2] for each in batch])
                next_states_batch = np.array([each[3] for each in batch])

                # Q values for the next_state, which is going to be our target Q
                target_Qs = sess.run(self.output, feed_dict={self.states: next_states_batch, self.is_train: True})
                episode_ends = (next_states_batch == np.zeros(current_state_batch[0].shape)).all(axis=(1, 2, 3))
                target_Qs[episode_ends] = (0, 0, 0)

                q_hat = rewards_batch + self.gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([self.loss, self.optm],
                                   feed_dict={self.states: current_state_batch,
                                              self.q_hat: q_hat,
                                              self.actions: actions_batch,
                                              self.is_train: True})
                step += 1

            print('Episode: {},'.format(episode),
                  'Total reward: {},'.format(total_reward),
                  'duration: {:.4f}'.format(time.time() - tic))
            summary_writer.add_summary(sess.run(summary_op, {total_rewards_ph: total_reward}), episode)
            # Start new episode
            self.initialize_game()

            if episode % 1 == 0:
                saver.save(sess, self.save_dir, episode)

    def test_deep_q_all(self):
        print('This method is not intended for uploading\n')
        self.config_test()
        restorer = tf.train.Saver()
        total_rewards_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('test/total_reward', total_rewards_ph)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.save_dir)
        sess = tf.Session()
        ckpt_list = self.list_getter(self.save_dir, 'index', 'model')
        for ckpt in ckpt_list:
            ckpt = ckpt.replace('.index', '')
            restorer.restore(sess, ckpt.replace('.index', ''))
            total_reward = 0
            for episode in range(self.num_repeat_episode):
                self.initialize_game()
                raw_screen = self.take_action(np.random.randint(0, self.num_actions), 0)
                frame = self.get_frame(raw_screen, self.state_shape)
                state_queue = deque([frame for _ in range(self.stack_frame_num)], maxlen=self.stack_frame_num)
                for step in range(self.max_step_per_episode):
                    state = np.stack(state_queue, axis=2)
                    action = np.argmax(sess.run(self.output, {self.states: np.expand_dims(state, axis=0), self.is_train: False}))
                    raw_screen = self.take_action(action, step)
                    frame = self.get_frame(raw_screen, self.state_shape)
                    state_queue.append(frame)
                    if self.is_collision:
                        total_reward -= 50
                        self.is_collision = False
                    else:
                        total_reward += 1
            ckpt_num = int(ckpt.split('-')[-1])
            print(ckpt_num)
            average_total_reward = float(total_reward) / float(self.num_repeat_episode)
            summary_writer.add_summary(sess.run(summary_op, {total_rewards_ph: average_total_reward}), ckpt_num)
        sess.close()

    def test_deep_q(self):
        self.config_test()
        restorer = tf.train.Saver()
        total_rewards_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('test/total_reward', total_rewards_ph)
        sess = tf.Session()
        ckpt = self.list_getter(self.save_dir, 'index', 'model')[-1]
        ckpt = ckpt.replace('.index', '')
        restorer.restore(sess, ckpt.replace('.index', ''))
        total_reward = 0
        for episode in range(self.num_repeat_episode):
            self.initialize_game()
            raw_screen = self.take_action(np.random.randint(0, self.num_actions), 0)
            frame = self.get_frame(raw_screen, self.state_shape)
            state_queue = deque([frame for _ in range(self.stack_frame_num)], maxlen=self.stack_frame_num)
            for step in range(self.max_step_per_episode):
                state = np.stack(state_queue, axis=2)
                action = np.argmax(sess.run(self.output, {self.states: np.expand_dims(state, axis=0), self.is_train: False}))
                raw_screen = self.take_action(action, step)
                frame = self.get_frame(raw_screen, self.state_shape)
                state_queue.append(frame)
                if self.is_collision:
                    total_reward -= 50
                    self.is_collision = False
                else:
                    total_reward += 1
        sess.close()
        average_total_reward = float(total_reward) / float(self.num_repeat_episode)
        print("average total reward: %d" % average_total_reward)


class PolicyGradient:
    def policy_gradient(self):
        raise NotImplemented()

    def train_policy_gradient(self):
        raise NotImplemented()

    def test_policy_gradient(self):
        raise NotImplemented()
