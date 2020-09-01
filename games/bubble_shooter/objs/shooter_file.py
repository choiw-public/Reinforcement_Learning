from games.bubble_shooter.objs.bubble_file import *
from math import sin, cos, radians
import pygame as pg


class Shooter:

    def __init__(self, image='./games/bubble_shooter/assets/gun.png', pos=display_rect.center):
        # center position of the image
        self.pos = pos
        self.pos_x, self.pos_y = pos
        self.width = self.pos_x
        self.height = self.pos_x

        self.initGunImage(image)

        self.angle = 90

        # Setup position of 'reloads'
        self.reload1_pos = (self.pos_x + 7 * BUBBLE_RADIUS, self.pos_y - 20)
        self.reload2_pos = (self.pos_x + 9.25 * BUBBLE_RADIUS, self.pos_y - 20)
        self.reload3_pos = (self.pos_x + 11.5 * BUBBLE_RADIUS, self.pos_y - 20)

        self.fired = Bullet(self.pos, self.angle)
        self.fired.exists = False
        self.loaded = Bubble(self.pos)
        self.reload1 = Bubble(self.reload1_pos)
        self.reload2 = Bubble(self.reload2_pos)
        self.reload3 = Bubble(self.reload3_pos)

    def initGunImage(self, image):
        # Load image
        self.shooter = pg.image.load(image).convert_alpha()

        # Get width and height
        self.shooter_rect = self.shooter.get_rect()
        self.shooter_w = self.shooter_rect[2]
        self.shooter_h = self.shooter_rect[3]

        # Scale image
        sf = 00.20
        self.shooter = pg.transform.scale(self.shooter, (int(self.shooter_w * sf), int(self.shooter_h * sf)))

        # Get new width and height
        self.shooter_rect = self.shooter.get_rect()
        self.shooter_w = self.shooter_rect[2]
        self.shooter_h = self.shooter_rect[3]

    # I could have put this in the initialization but I wanted to emphasize the fact that the image we are actually rotating is in a box
    def putInBox(self):

        # Make a box to put shooter in
        # Surface((width, height), flags=0, depth=0, masks=None) -> Surface
        self.shooter_box = pg.Surface((self.shooter_w, self.shooter_h * 2), pg.SRCALPHA, 32)
        self.shooter_box.fill((0, 0, 0, 0))

        # Put shooter in box
        self.shooter_box.blit(self.shooter, (0, 0))

        # Since we want 90 to be when the shooter is pointing straight up, we rotate it
        self.shooter_box = pg.transform.rotate(self.shooter_box, -90)

    def rotate(self, x):
        # Get angle of rotation (in degrees)
        # self.angle = self.get_angle(x)
        self.angle = x

        # Get a rotated version of the box to display. Note: don't keep rotating the original as that skews the image
        rotated_box = pg.transform.rotate(self.shooter_box, x)

        # display the image
        display.blit(rotated_box, rotated_box.get_rect(center=self.pos))

    def draw_bullets(self):

        self.fired.update()
        self.loaded.draw()
        self.reload1.draw()
        self.reload2.draw()
        self.reload3.draw()

    def fire(self):

        if self.fired.exists:
            return

        else:
            rads = radians(self.angle)
            self.fired = Bullet(self.pos, rads, self.loaded.color)
            self.loaded = Bubble(self.pos, self.reload1.color)
            self.reload1 = Bubble(self.reload1_pos, self.reload2.color)
            self.reload2 = Bubble(self.reload2_pos, self.reload3.color)
            self.reload3 = Bubble(self.reload3_pos)

    def get_angle(self, x):
        self.angle = x
        # Restrict the angles, we don't want the user to be able to point all the way
        return max(min(self.angle, ANGLE_MAX), ANGLE_MIN)
