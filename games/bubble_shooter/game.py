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

    def __init__(self):
        self.vis = True
        self.frame_count = 0

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
        # display.fill((255, 255, 255))
        # self.grid_manager.view(self.gun, self.game)  # Check collision with bullet and update grid as needed
        # self.gun.draw_bullets()  # Draw and update bullet and reloads
        return self.get_state()

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

    def get_state(self):
        checkboard = self.grid_manager.state
        checkboard = np.repeat(checkboard, 2, 0)[:-1, ::]
        masks = []
        for bc in BUBBLE_COLORS:
            lower = np.array(bc) - 1
            upper = np.array(bc) + 1
            mask = cv.inRange(checkboard, lower, upper)
            mask[mask > 0] = 1
            masks.append(mask)

        bullets = [self.gun.loaded.color, self.gun.reload1.color, self.gun.reload2.color, self.gun.reload3.color]
        state = []
        for bc in bullets[:3]:
            index = BUBBLE_COLORS.index(bc)
            main = masks[index]
            others = np.sum(masks[:index] + masks[index + 1:], axis=0)
            state.append(np.stack([main, others], 2))
        return np.concatenate(state, 2)

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
            reward = self.grid_manager.reward
            if self.game.over:
                reward = -50
            elif np.sum(self.grid_manager.state) == 0:
                reward = 100
                pygame.display.update()
                print('WOW!!! You Won')
                return self.get_state(), True, reward
            pygame.display.update()
        return self.get_state(), self.game.over, reward
