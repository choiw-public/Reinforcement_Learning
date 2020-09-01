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

        self.gun.rotate(90)  # Rotate the gun if the mouse is moved
        self.gun.draw_bullets()  # Draw and update bullet and reloads
        self.game.drawScore()  # draw score
        pygame.display.update()
        return self.image_to_state()

    def manual_play(self):
        raise ValueError("re-configure this method.")
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
        img_raw = np.transpose(img_raw, [1, 0, 2]) / 255.0
        h, w, _ = img_raw.shape
        scale = float(self.state_size) / float(max(h, w))
        return cv.resize(img_raw, (int(w * scale), int(h * scale)))

    def take_action(self, angle):
        self.background.draw()  # Draw BG first
        self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed
        self.gun.rotate(angle)  # Rotate the gun if the mouse is moved
        self.gun.fire()
        self.gun.draw_bullets()  # Draw and update bullet and reloads
        self.game.drawScore()  # draw score
        pygame.display.update()
        while self.gun.fired.exists:
            self.background.draw()  # Draw BG first
            self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed
            self.gun.rotate(angle)  # Rotate the gun if the mouse is moved
            self.gun.draw_bullets()  # Draw and update bullet and reloads
            self.game.drawScore()  # draw score
            pygame.display.update()
        return self.image_to_state(), self.game.over, self.grid_manager.reward

    def take_action_backup(self, angle):
        self.background.draw()  # Draw BG first
        self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed
        self.gun.rotate(angle)  # Rotate the gun if the mouse is moved
        self.gun.fire()
        self.gun.draw_bullets()  # Draw and update bullet and reloads
        self.game.drawScore()  # draw score
        pygame.display.update()
        return self.image_to_state(), self.game.over, self.grid_manager.reward
