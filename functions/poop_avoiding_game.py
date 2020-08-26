import random
import pygame
import numpy as np


class Character(pygame.sprite.Sprite):
    def __init__(self):
        super(Character, self).__init__()
        self.images = {'right':
                           {'run': [pygame.image.load('./asset/char/run_r/%d.png' % i) for i in range(6)],
                            'idle': [pygame.image.load('./asset/char/idle_r/%d.png' % i) for i in range(4)]},
                       'left':
                           {'run': [pygame.image.load('./asset/char/run_l/%d.png' % i) for i in range(6)],
                            'idle': [pygame.image.load('./asset/char/idle_l/%d.png' % i) for i in range(4)]}}
        # index [right_run, right_idle, left_run, left_idle]
        self.index = [0, 0, 0, 0]
        self.index_max = [len(self.images['right']['run']),
                          len(self.images['right']['idle']),
                          len(self.images['left']['run']),
                          len(self.images['left']['idle'])]
        self.direction_prev = 'right'
        self.action_prev = 0
        self.size = self.images['right']['run'][0].get_rect().size

    def image(self, action=0):
        if action != self.action_prev:
            self.index = [0, 0, 0, 0]
        if action == 0:  # if aciton is idle
            if self.direction_prev == 'right':  # and the previous direction was right
                img = self.images['right']['idle'][self.index[1]]
                self.index[1] += 1
                self.direction_prev = 'right'
            else:
                img = self.images['left']['idle'][self.index[3]]
                self.index[3] += 1
                self.direction_prev = 'left'
        elif action == 1:  # if action is run left
            img = self.images['left']['run'][self.index[2]]
            self.index[2] += 1
            self.direction_prev = 'left'
        elif action == 2:  # if action is run right
            img = self.images['right']['run'][self.index[0]]
            self.index[0] += 1
            self.direction_prev = 'right'
        self.action_prev = action
        # reset indices if over max
        for _ in range(len(self.index)):
            if self.index[_] >= self.index_max[_]:
                self.index[_] = 0
        return img


class PoopAvoiding:
    """
    The main idea of developing this game is from:
    https://nadocoding.tistory.com/8
    """

    def initialize_game(self):
        pygame.init()
        self.screen_width = 480
        self.screen_height = 640
        pygame.display.set_caption("Dodge the poops!!")
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
        self.poop_image = "./asset/poop/poop.png"
        self.poop = [pygame.image.load(self.poop_image)]
        self.poop_size = self.poop[0].get_rect().size
        self.poop_width = self.poop_size[0]
        self.poop_height = self.poop_size[1]
        self.poop_x_pos = [np.random.randint(0, self.screen_width - self.poop_width)]
        self.poop_y_pos = [0]
        self.poop_speed = 10
        self.poop_x_hist = [(self.poop_x_pos[0] + self.poop_width * 0.5) / self.screen_width]
        self.max_poop_num = 30
        self.poop_add_interval = int(self.screen_height / self.poop_speed / self.max_poop_num)
        self.poop_add_prob = 0.5
        self.clock = pygame.time.Clock()
        self.is_collision = False
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.update_display()
        return pygame.surfarray.array2d(self.screen)

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
            self.clock.tick(30)
            self.take_action(action, step)
            step += 1

    def update_display(self, action=0):
        # self.screen.blit(self.background, (0, 0))
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.character.image(action), (self.character_x_pos, self.character_y_pos))
        for poop, poop_x, poop_y in zip(self.poop, self.poop_x_pos, self.poop_y_pos):
            self.screen.blit(poop, (poop_x, poop_y))
        pygame.display.update()

    def take_action(self, action, step):
        # m = self.character_move * self.character_momentum
        self.character_move *= self.character_momentum
        if action == 0:  # this will be stay
            # self.character_move = m
            pass
        elif action == 1:  # move left
            self.character_move -= self.character_speed
        elif action == 2:  # turn this into 1 for left 2 for right
            self.character_move += self.character_speed
        # self.character_move += m
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
        character_rect.left = self.character_x_pos
        character_rect.top = self.character_y_pos
        character_rect.w = character_rect.w * 0.5
        character_rect.h = character_rect.h * 0.5

        # handle poops
        poop_idx_to_remove = []
        for poop_idx in range(len(self.poop_y_pos)):
            self.poop_y_pos[poop_idx] += self.poop_speed
            poop_rect = self.poop[poop_idx].get_rect()
            poop_rect.left = self.poop_x_pos[poop_idx]
            poop_rect.top = self.poop_y_pos[poop_idx]
            poop_rect.w = poop_rect.w * 0.5
            poop_rect.h = poop_rect.h * 0.5
            if character_rect.colliderect(poop_rect):
                self.is_collision = True
                poop_idx_to_remove.append(poop_idx)
            # if a poop is gone remove the poop from screen
            if self.poop_y_pos[poop_idx] > self.screen_height:
                poop_idx_to_remove.append(poop_idx)
        self.poop = [v for _, v in enumerate(self.poop) if _ not in poop_idx_to_remove]
        self.poop_x_pos = [v for _, v in enumerate(self.poop_x_pos) if _ not in poop_idx_to_remove]
        self.poop_y_pos = [v for _, v in enumerate(self.poop_y_pos) if _ not in poop_idx_to_remove]

        # add poop by interval
        if step != 0 and step % self.poop_add_interval == 0 and self.poop_add_prob > np.random.uniform(0, 1):
            self.poop.append(pygame.image.load(self.poop_image))
            if np.random.uniform(0, 1) > 0.9:
                self.poop_x_pos.append(self.character_x_pos)
            else:
                self.poop_x_pos.append(random.randint(0, self.screen_width - self.poop_width))
            self.poop_y_pos.append(0)
            self.poop_x_hist.append((self.poop_x_pos[-1] + self.poop_width * 0.5) / self.screen_width)
        self.update_display(action)
        return pygame.surfarray.array3d(self.screen)
