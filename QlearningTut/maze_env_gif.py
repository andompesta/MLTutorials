"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys
from PIL import Image, ImageDraw, ImageFont


UNIT = 40  # pixels
DISCOUNT_BAD_ACTION = 0
GOAL_REWARD = 1
HOLES_REWARD = -1

class EnvObj():
    def __init__(self, coords, reward, name):
        self.x = coords[0]
        self.y = coords[1]
        self.reward = reward
        self.name = name


class Maze():
    def __init__(self, height=4, width=4):
        self.maze_h = height
        self.maze_w = width
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.obj = {}
        self._build_maze()

    def _build_maze(self):
        '''
        Bluind the state space
        :return:
        '''

        for idx, x_coord in enumerate(range(1, self.maze_w - 1)):
            self.obj["hell{}".format(idx)] = EnvObj((x_coord, 0), HOLES_REWARD, "hell{}".format(idx))

        self.obj["goal"] = EnvObj((self.maze_w-1, 0), GOAL_REWARD, "goal")
        self.reset()

    def reset(self):
        """
        Create a new starting point
        :return:
        """
        self.obj["hero"] = EnvObj((0, 0), 0, "hero")
        self.block_position = []
        # self.block_position = [[value.x, value.y] for key, value in self.obj.items() if value.reward == -1]
        return [0, 0]

    def step(self, action):
        '''
        execute one step
        :param action: action to execute
        :return: env, env_image, reward obtained, position of the goal, position of the hero
        '''
        s_, bad_action_reward = self.move_hero(action)     # move the hero
        goal_reward, done = self.check_goal(s_)
        return s_, bad_action_reward+goal_reward, done

    def move_hero(self, action):
        '''
        there are 6 actions:
        - 0 - up
        - 1 - down
        - 2 - left
        - 3 - right
        :param action: action to take
        :return: new position of the drone according to the action done, the penality encountered during the action taken
        '''
        hero = self.obj["hero"]
        bad_action_reward = 0

        if action < self.n_actions:
            if action == 0 and hero.y >= 1 and [hero.x, hero.y - 1] not in self.block_position:
                hero.y -= 1
            elif action == 0 and hero.y == 0:
                bad_action_reward = DISCOUNT_BAD_ACTION

            if action == 1 and hero.y <= self.maze_h-2 and [hero.x, hero.y + 1] not in self.block_position:
                hero.y += 1
            elif action == 1 and hero.y == self.maze_h-1:
                bad_action_reward = DISCOUNT_BAD_ACTION

            if action == 2 and hero.x >= 1 and [hero.x - 1, hero.y] not in self.block_position:
                hero.x -= 1
            elif action == 2 and hero.x == 0:
                bad_action_reward = DISCOUNT_BAD_ACTION

            if action == 3 and hero.x <= self.maze_w-2 and [hero.x + 1, hero.y] not in self.block_position:
                hero.x += 1
            elif action == 3 and hero.x == self.maze_w-1:
                bad_action_reward = DISCOUNT_BAD_ACTION

        if self.obj["hero"].x == hero.x and self.obj["hero"].y == hero.y:
            bad_action_reward = DISCOUNT_BAD_ACTION

        self.obj["hero"] = hero
        return [hero.x, hero.y], bad_action_reward


    def check_goal(self, hero_coord):
        '''
        check if we get some rewards
        :return: reward obtained, a boolean
        '''
        goal_coord = [self.obj["goal"].x, self.obj["goal"].y]
        hell_coords = [[value.x, value.y] for key, value in self.obj.items() if "hell" in value.name]

        if hero_coord == goal_coord:
            return self.obj["goal"].reward, True

        if hero_coord in hell_coords:
            return self.obj["hell0"].reward, True

        return 0, False

    def render(self, step=0):
        img = np.ones([(self.maze_h * UNIT) + 40, self.maze_w * UNIT, 3]) * 255.0
        img = Image.fromarray(img.astype("uint8"))
        draw = ImageDraw.Draw(img)

        # create grids
        for c in range(UNIT, self.maze_w * UNIT, UNIT):
            draw.line([(c, 0), (c, self.maze_h * UNIT)], fill=(128, 128, 128))
        for r in range(UNIT, (self.maze_h+1) * UNIT, UNIT):
            draw.line([(0, r), (self.maze_w * UNIT, r)], fill=(128, 128, 128))

        # create goal
        draw.ellipse([((self.obj["goal"].x * UNIT) + 5, (self.obj["goal"].y * UNIT) + 5),
                      ((self.obj["goal"].x * UNIT) + UNIT - 5, (self.obj["goal"].y * UNIT) + UNIT - 5)],
                     fill=(250, 255, 0))

        # create hell
        for idx in range(self.maze_w - 2):
            draw.rectangle(
                [((self.obj["hell{}".format(idx)].x * UNIT) + 5, (self.obj["hell{}".format(idx)].y * UNIT) + 5),
                 ((self.obj["hell{}".format(idx)].x * UNIT) + UNIT - 5,
                  (self.obj["hell{}".format(idx)].y * UNIT) + UNIT - 5)],
                fill=(0, 0, 0))

        # create hero
        draw.rectangle([((self.obj["hero"].x * UNIT) + 5, (self.obj["hero"].y * UNIT) + 5),
                        ((self.obj["hero"].x * UNIT) + UNIT - 5, (self.obj["hero"].y * UNIT) + UNIT - 5)],
                       fill=(255, 0, 0))

        font = ImageFont.truetype("./image/FreeSans.ttf", 18)
        draw.text((140, 170), 'Step: ' + str(step), (0, 0, 0), font=font)

        return img

if __name__ == "__main__":
    env = Maze(height=4, width=8)
    img = env.render()
    img.save("env.png", format="png")
