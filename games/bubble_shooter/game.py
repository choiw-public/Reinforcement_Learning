from games.bubble_shooter.objs.grid_file import *
from games.bubble_shooter.objs.shooter_file import *
from games.bubble_shooter.objs.game_objects import *
import pygame
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

pygame.init()


class BubbleShooter:
    """
    This game is from:https://github.com/PranavB6/Bubbles_2.0
    """

    def __init__(self, state_size):
        self.state_size = state_size

    def initialize_game(self):
        # Create background
        self.background = Background()

        # Initialize gun, position at bottom center of the screen
        self.gun = Shooter(pos=BOTTOM_CENTER)
        self.gun.putInBox()

        self.grid_manager = GridManager()
        self.game = Game()

        self.background.draw()  # Draw BG first

        self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed

        self.gun.rotate(random.randint(-2, 2))  # Rotate the gun if the mouse is moved
        self.gun.draw_bullets()  # Draw and update bullet and reloads
        self.game.drawScore()  # draw score
        self.angle_move = 0
        pygame.display.update()

    def manual_play(self):
        self.initialize_game()
        while not self.game.over:
            clock.tick(60)  # 60 FPS
            # quit when you press the x
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.angle_move += 1
                    elif event.key == pygame.K_RIGHT:
                        self.angle_move -= 1
                    elif event.key == pygame.K_SPACE:
                        self.gun.fire()
                else:
                    self.angle_move = 0
            self.background.draw()  # Draw BG first
            self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed
            self.gun.rotate(self.angle_move)  # Rotate the gun if the mouse is moved
            self.gun.draw_bullets()  # Draw and update bullet and reloads
            self.game.drawScore()  # draw score
            if self.game.over:
                self.initialize_game()
            pygame.display.update()

    def image_to_state(self):
        # # normalize screen image and resize to a small image
        # # the resized image is considered as state
        # # Note: "state_size" [height, width], but opencv accept [width, and height]
        img_raw = pygame.surfarray.array3d(display)[WALL_BOUND_L:WALL_BOUND_R, :, :]
        img_raw = cv.cvtColor(np.transpose(img_raw, [1, 0, 2]), cv.COLOR_RGB2GRAY) / 255.0
        # crop raw_screen from bottom so that the square image
        h, w = img_raw.shape
        scale = float(self.state_size) / float(max(h, w))
        return cv.resize(img_raw, (int(w * scale), int(h * scale)))

    def take_action(self, action):
        if action == 0:  # stay
            self.angle_move = 0
        elif action == 1:  # move left
            self.angle_move += 1
        elif action == 2:  # move right
            self.angle_move -= 1
        elif action == 3:
            self.angle_move = 0
        elif action == 4:
            self.gun.fire()
        self.background.draw()  # Draw BG first
        self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed
        self.gun.rotate(self.angle_move)  # Rotate the gun if the mouse is moved
        self.gun.draw_bullets()  # Draw and update bullet and reloads
        self.game.drawScore()  # draw score
        pygame.display.update()
        return self.image_to_state(), self.game.over, self.game.reward
