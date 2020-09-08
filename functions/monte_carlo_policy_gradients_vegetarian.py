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
        self.logit = tf.layers.dense(net, num_actions)
        self.action_prob = tf.nn.softmax(self.logit)


class VegetarianPolicyGradient(CNN, AdventurerTheVegetarian):
    def __init__(self, config):
        if config.phase == 'manual_play':
            self.manual_play()
        self.state_stack_num = 4
        self.state_size = 65
        self.policy_gradient()
        if config.phase == 'train':
            self.train_monte_carlo_policy_gradient()
        elif config.phase == 'test':
            self.test_deep_q()
        else:
            raise ValueError('Unexpected phase')

    def config_train(self):
        self.max_epoch = 10000
        self.batch_size = 2000
        self.gamma = 0.99  # future reward discount
        self.lr = 0.001  # learning rate
        self.save_dir = './checkpoints/vegetarian/Policy_Gradient'
        self.save_interval = 20

    def config_test(self):
        self.max_step_per_episode = 2000
        self.num_repeat_episode = 10
        self.save_dir = './checkpoints/vegetarian/Policy_Gradient'

    def policy_gradient(self, num_actions=3):
        self.state = tf.placeholder(tf.float32, [None, self.state_size, self.state_size, self.state_stack_num])
        self.action = tf.placeholder(tf.int32, [None])
        self.discounted_reward = tf.placeholder(tf.float32, [None])
        self.is_train = tf.placeholder(tf.bool, None)
        self.num_actions = num_actions

    def get_discounted_reward(self, reward_record):
        discounted_episode_rewards = np.zeros_like(reward_record)
        cumulative = 0.0
        for i in reversed(range(len(reward_record))):
            cumulative = cumulative * self.gamma + reward_record[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        return list((discounted_episode_rewards - mean) / (std))

    def get_batch(self, sess):
        # frame: a single channel image converted from raw_screen
        # state_queue: a queue to enqueue and dequeue
        # current_state & next_state: h x w x state_stack_num

        # Initialize lists: states, action, rewards_of_episode, rewards_of_batch, discounted_rewards
        state_batch = []
        action_batch = []
        reward_in_episode = []
        reward_batch = []
        discounted_reward_batch = []
        episode_count = 1

        self.initialize_game()
        state_queue = deque(maxlen=self.state_stack_num)

        # iterate to get very first state
        for i in range(self.state_stack_num):
            frame, _, _ = self.take_action(0)
            state_queue.append(frame)
        current_state = np.stack(state_queue, axis=2)

        while True:
            feed_dict = {self.state: np.expand_dims(current_state, 0),
                         self.is_train: False}
            action_prob = sess.run(self.action_prob, feed_dict)

            # select action by action probability
            action = np.random.choice(range(self.num_actions), p=action_prob.ravel())
            next_frame, is_end, reward = self.take_action(action)

            # append to batch
            state_batch.append(current_state)
            action_batch.append(action)

            # record rewards and discount it later when is_end = True
            reward_in_episode.append(reward)

            if is_end:
                reward_batch.append(np.sum(reward_in_episode))

                discounted_reward = self.get_discounted_reward(reward_in_episode)
                discounted_reward_batch += discounted_reward

                if len(state_batch) > self.batch_size:
                    break

                reward_in_episode = []
                episode_count += 1
                self.initialize_game()
                state_queue = deque(maxlen=self.state_stack_num)

                # iterate to fill get very first state
                for i in range(self.state_stack_num):
                    frame, _, _ = self.take_action(0)
                    state_queue.append(frame)
                current_state = np.stack(state_queue, axis=2)
            else:
                state_queue.append(next_frame)
                next_state = np.stack(state_queue, axis=2)
                current_state = next_state

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        average_reward_per_episode = np.average(reward_batch)
        discounted_reward_batch = np.array(discounted_reward_batch)
        return state_batch, action_batch, average_reward_per_episode, discounted_reward_batch, episode_count

    def train_monte_carlo_policy_gradient(self):
        self.config_train()
        self.build_cnn(self.num_actions)
        onehot_action = tf.one_hot(self.action, self.num_actions)
        negative_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logit, labels=onehot_action)
        self.loss = tf.reduce_mean(negative_log_prob * self.discounted_reward)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optm = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.optm = tf.group([optm, update_ops])
        saver = tf.train.Saver(max_to_keep=1000)
        average_reward_per_episode_ph = tf.placeholder(tf.int64)
        tf.summary.scalar('train/reward_gain', average_reward_per_episode_ph)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.save_dir)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch_num in range(1, self.max_epoch):
            state_batch, action_batch, avg_reward_per_ep, discounted_reward_batch, episode_count = self.get_batch(sess)

            feed_dict = {self.state: state_batch,
                         self.action: action_batch,
                         self.discounted_reward: discounted_reward_batch,
                         self.is_train: True}
            loss, _ = sess.run([self.loss, self.optm], feed_dict)

            print('Episode: {},'.format(epoch_num),
                  'Average reward: {:.4f}'.format(avg_reward_per_ep))
            summary_writer.add_summary(sess.run(summary_op, {average_reward_per_episode_ph: avg_reward_per_ep}), epoch_num)
            if epoch_num % self.save_interval == 0:
                saver.save(sess, self.save_dir + '/model', epoch_num)

    def test_deep_q_all(self):
        raise NotImplemented('this method should be modified')
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
                frame, _, _ = self.take_action(np.random.randint(0, self.num_actions), 0)
                state_queue = deque([frame for _ in range(self.state_stack_num)], maxlen=self.state_stack_num)
                for step in range(self.max_step_per_episode):
                    state = np.stack(state_queue, axis=2)
                    action = np.argmax(sess.run(self.prob_distribution, {self.state: np.expand_dims(state, axis=0), self.is_train: False}))
                    if is_end:
                        break
                    frame, is_end, reward = self.take_action(action, step)
                    state_queue.append(frame)
                    total_reward += reward
            ckpt_num = int(ckpt.split('-')[-1])
            print(ckpt_num)
            average_total_reward = float(total_reward) / float(self.num_repeat_episode)
            summary_writer.add_summary(sess.run(summary_op, {total_rewards_ph: average_total_reward}), ckpt_num)
        sess.close()

    def test_deep_q(self):
        raise NotImplemented('this method should be modified')
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
            frame, _, _ = self.take_action(np.random.randint(0, self.num_actions), 0)
            state_queue = deque([frame for _ in range(self.state_stack_num)], maxlen=self.state_stack_num)
            for step in range(self.max_step_per_episode):
                state = np.stack(state_queue, axis=2)
                action = np.argmax(sess.run(self.prob_distribution, {self.state: np.expand_dims(state, axis=0), self.is_train: False}))
                frame, is_end, reward = self.take_action(action, step)
                # if is_end:
                #     break
                state_queue.append(frame)
                total_reward += reward
        sess.close()
        average_total_reward = float(total_reward) / float(self.num_repeat_episode)
        print("average total reward: %d" % average_total_reward)
