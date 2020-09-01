import random
import pygame
import numpy as np
import cv2 as cv


class Character:
    """
    character is from:
    https://rvros.itch.io/animated-pixel-hero
    """

    def __init__(self):
        super(Character, self).__init__()
        base_dir = './games/adventurer_the_vegetarian/asset/char'
        self.images = {'right':
                           {'run': [pygame.image.load(base_dir + '/run_r/%d.png' % i) for i in range(6)],
                            'idle': [pygame.image.load(base_dir + '/idle_r/%d.png' % i) for i in range(4)]},
                       'left':
                           {'run': [pygame.image.load(base_dir + '/run_l/%d.png' % i) for i in range(6)],
                            'idle': [pygame.image.load(base_dir + '/idle_l/%d.png' % i) for i in range(4)]}}
        # index [right_run, right_idle, left_run, left_idle]
        self.index = [0, 0, 0, 0]
        self.frame_delay = 2
        self.index_max = [len(self.images['right']['run']) * self.frame_delay,
                          len(self.images['right']['idle'] * self.frame_delay),
                          len(self.images['left']['run'] * self.frame_delay),
                          len(self.images['left']['idle']) * self.frame_delay]
        self.direction_prev = 'right'
        self.action_prev = 0
        self.size = self.images['right']['run'][0].get_rect().size

    def image(self, action=0):
        if action != self.action_prev:
            self.index = [0, 0, 0, 0]
        if action == 0:  # if aciton is idle
            if self.direction_prev == 'right':  # and the previous direction was right
                img = self.images['right']['idle'][self.index[1] // self.frame_delay]
                self.index[1] += 1
                self.direction_prev = 'right'
            else:
                img = self.images['left']['idle'][self.index[3] // self.frame_delay]
                self.index[3] += 1
                self.direction_prev = 'left'
        elif action == 1:  # if action is run left
            img = self.images['left']['run'][self.index[2] // self.frame_delay]
            self.index[2] += 1
            self.direction_prev = 'left'
        elif action == 2:  # if action is run right
            img = self.images['right']['run'][self.index[0] // self.frame_delay]
            self.index[0] += 1
            self.direction_prev = 'right'
        self.action_prev = action
        # reset indices if over max
        for _ in range(len(self.index)):
            if self.index[_] >= self.index_max[_]:
                self.index[_] = 0
        return img


class HorribleMeat:
    """
    meat images are from :https://vectorpixelstar.itch.io/food?download
    """

    def __init__(self):
        base_dir = './games/adventurer_the_vegetarian/asset/meat'
        self.meat_images = [base_dir + "/%d.png" % i for i in range(6)]
        self.img, self.width, self.height, self.speed = [], [], [], []
        self.add_meat()

    def add_meat(self):
        self.img.append(pygame.image.load(self.meat_images[random.randint(0, 5)]))
        size = self.img[-1].get_rect().size
        self.width.append(size[0])
        self.height.append(size[1])
        self.speed.append(random.randint(8, 12))


class AdventurerTheVegetarian:
    """
    The Handler idea of developing this game is from:
    https://nadocoding.tistory.com/8
    """

    def initialize_game(self):
        pygame.init()
        self.screen_width = 480
        self.screen_height = 640
        pygame.display.set_caption("Adventurer: the vegetarian!")
        self.clock = pygame.time.Clock()
        self.character = Character()
        self.character_width = self.character.size[0]
        self.character_height = self.character.size[1]
        self.character_x_pos = np.random.randint(0, self.screen_width - self.character_width)
        self.character_y_pos = self.screen_height - self.character_height
        self.character_speed = 3
        self.character_momentum = 0.9
        self.character_move = 0
        self.character_move_limit = 10
        self.meat = HorribleMeat()
        self.meat_add_interval = 2
        self.meat_add_prob = 0.1
        self.meat_x_pos = [np.random.randint(0, self.screen_width - self.meat.width[0])]
        self.meat_y_pos = [0]
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.update_display()

    def manual_play(self):
        self.initialize_game()
        step = 1
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:  # this is unnecessary as I will divide 0, 1, 2 for stay, left, right
                    if event.key == pygame.K_LEFT:  # turn this into 1 for left 2 for right
                        action = 1
                    elif event.key == pygame.K_RIGHT:  # turn this into 1 for left 2 for right
                        action = 2
                else:
                    action = 0
            self.clock.tick(50)
            self.take_action(action, step, manual=True)
            step += 1

    def update_display(self, action=0):
        # self.screen.blit(self.background, (0, 0))
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.character.image(action), (self.character_x_pos, self.character_y_pos))
        for meat, meat_x, meat_y in zip(self.meat.img, self.meat_x_pos, self.meat_y_pos):
            self.screen.blit(meat, (meat_x, meat_y))
        pygame.display.update()

    def image_to_state(self, step):
        # # normalize screen image and resize to a small image
        # # the resized image is considered as state
        # # Note: "state_size" [height, width], but opencv accept [width, and height]
        img_raw = pygame.surfarray.array3d(self.screen)

        tmp = cv.resize(img_raw, dsize=(320, 240))
        tmp = np.pad(tmp, [[5, 5], [5, 5], [0, 0]])
        cv.imwrite('./demo/private_adventurer/raw_frames/frame%04d.jpg' % step, np.transpose(tmp[:, :, ::-1], [1, 0, 2]))

        img_raw = cv.cvtColor(np.transpose(img_raw, [1, 0, 2]), cv.COLOR_RGB2GRAY) / 255.0

        h, w = img_raw.shape
        scale = float(self.state_size) / float(w)
        frame = np.squeeze(cv.resize(img_raw, (self.state_size, self.state_size)))
        pad_length = int(self.state_size * 0.6)
        frame = np.pad(frame, [[0, 0], [pad_length, pad_length]])
        character_center = int((self.character_x_pos + self.character_width * 0.5) * scale) + pad_length
        x = character_center - int(self.state_size * 0.5)
        return frame[:, x:x + self.state_size]

    def take_action(self, action, step, manual=False):
        self.character_move *= self.character_momentum
        if action == 0:  # stay
            pass
        elif action == 1:  # move left
            self.character_move -= self.character_speed
        elif action == 2:  # move right
            self.character_move += self.character_speed
        if self.character_move > self.character_move_limit:
            self.character_move = self.character_move_limit
        elif self.character_move < -self.character_move_limit:
            self.character_move = -self.character_move_limit
        self.character_x_pos += self.character_move

        if self.character_x_pos < 0:
            self.character_x_pos = 0
        elif self.character_x_pos > self.screen_width - self.character_width:
            self.character_x_pos = self.screen_width - self.character_width

        character_rect = self.character.images['right']['run'][0].get_rect()
        character_rect.left = self.character_x_pos + character_rect.w * 0.5 * 0.25
        character_rect.top = self.character_y_pos
        character_rect.h = character_rect.h * 0.5
        character_rect.w = character_rect.w * 0.5

        # handle meats
        meat_idx_to_remove = []
        self.is_end = False
        for meat_idx in range(len(self.meat_y_pos)):
            self.meat_y_pos[meat_idx] += self.meat.speed[meat_idx]
            meat_rect = self.meat.img[meat_idx].get_rect()
            meat_rect.left = self.meat_x_pos[meat_idx]
            meat_rect.top = self.meat_y_pos[meat_idx] - self.meat.height[meat_idx] * 0.25
            meat_rect.w = meat_rect.w * 0.6
            meat_rect.h = meat_rect.h
            if character_rect.colliderect(meat_rect):
                meat_idx_to_remove.append(meat_idx)
                self.is_end = True
                self.reward = -5
            else:
                x = self.character_x_pos + character_rect.w * 0.5
                self.reward = 1.0 / (1.0 + abs((x - 230.0) / 140) ** (2 * 3))
            # remove a meat from screen if it's out of screen
            if self.meat_y_pos[meat_idx] > self.screen_height:
                meat_idx_to_remove.append(meat_idx)
        self.meat.img = [v for _, v in enumerate(self.meat.img) if _ not in meat_idx_to_remove]
        self.meat.width = [v for _, v in enumerate(self.meat.width) if _ not in meat_idx_to_remove]
        self.meat.height = [v for _, v in enumerate(self.meat.height) if _ not in meat_idx_to_remove]
        self.meat.speed = [v for _, v in enumerate(self.meat.speed) if _ not in meat_idx_to_remove]
        self.meat_x_pos = [v for _, v in enumerate(self.meat_x_pos) if _ not in meat_idx_to_remove]
        self.meat_y_pos = [v for _, v in enumerate(self.meat_y_pos) if _ not in meat_idx_to_remove]

        # add meat by interval
        self.meat_add_prob += 0.0009
        if step != 0 and step % self.meat_add_interval == 0 and self.meat_add_prob > np.random.uniform(0, 1):
            self.meat.add_meat()

            if np.random.uniform(0, 1) > 0.8:
                self.meat_x_pos.append(self.character_x_pos)
            else:
                # use beta distribution
                self.meat_x_pos.append(max(0, int(np.random.beta(0.8, 0.8) * self.screen_width) - self.meat.width[-1]))
            self.meat_y_pos.append(0)
        self.update_display(action)
        if not manual:
            return self.image_to_state(step)
