from collections import deque
from functions.poop_avoiding_game import PoopAvoiding
from functions.reinforcement_learning import CNN, DeepQ, PolicyGradient
import numpy as np
import cv2 as cv
import os
import re


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
        self.count -= batch_size
        return [self.buffer[ii] for ii in idx]


class Handler(PoopAvoiding, CNN, DeepQ, PolicyGradient):
    def __init__(self, config):
        if config.phase == 'manual_play':
            self.manual_play()
        if config.RL_type == 'deep_q':
            self.stack_frame_num = 4
            self.state_shape = [65, 65]  # [32, 24]
            self.deep_q()
            if config.phase == 'train':
                self.state_queue = Queue()
                self.fill_initial_states(max_length_per_episode=100)
                self.train_deep_q()
            elif config.phase == 'test':
                self.test_deep_q()
            else:
                raise ValueError('Unexpected phase')
        else:
            raise NotImplemented()

    @staticmethod
    def get_frame(raw_screen, state_shape):
        # normalize screen image and resize to a small image
        # the resized image is considered as state
        # Note: "state_shape" [height, width], but opencv accept [width, and height]
        raw_screen = np.average(np.transpose(raw_screen, [1, 0, 2]) / 255.0, 2)
        return np.squeeze(cv.resize(raw_screen, (state_shape[1], state_shape[0])))

    @staticmethod
    def list_getter(dir_name, extension, must_include=None):
        def sort_nicely(a_list):
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            a_list.sort(key=alphanum_key)

        file_list = []
        if dir_name:
            for path, subdirs, files in os.walk(dir_name):
                for name in files:
                    if name.lower().endswith((extension)):
                        if must_include:
                            if must_include in name:
                                file_list.append(os.path.join(path, name))
                        else:
                            file_list.append(os.path.join(path, name))
            sort_nicely(file_list)
        return file_list

    def fill_initial_states(self, init_pop_num=100, max_length_per_episode=500):
        # raw_screen: capture of the game screen
        # frame: a single channel image converted from raw_screen
        # state_queue: a queue to enqueue and dequeue
        # current_state & next_state: h x w x stack_frame_num

        self.initialize_game()
        raw_screen = self.take_action(np.random.randint(0, 3), 0)

        frame = self.get_frame(raw_screen, self.state_shape)

        # duplicate the frame by "stack_frame_num" times
        state_queue = deque([frame for _ in range(self.stack_frame_num)], maxlen=self.stack_frame_num)
        while self.state_queue.count < init_pop_num:
            for step in range(1, max_length_per_episode):
                random_action = np.random.randint(0, self.num_actions)
                raw_screen = self.take_action(random_action, step)  # this is next_state

                if self.is_collision:
                    # collision, so no next_state
                    next_state = np.zeros(current_state.shape)
                    # below, reward -10. - because collided
                    self.state_queue.add((current_state, random_action, -10, next_state))
                    self.initialize_game()
                else:
                    current_state = np.stack(state_queue, axis=2)

                    next_frame = self.get_frame(raw_screen, self.state_shape)
                    state_queue.append(next_frame)
                    next_state = np.stack(state_queue, axis=2)

                    self.state_queue.add((current_state, random_action, 1, next_state))
                    current_state = next_state
        print('initial Queue is filled')
