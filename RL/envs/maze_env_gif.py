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
import torchvision.transforms as T
from gym import error, spaces, utils, Env

UNIT = 40  # pixels
DISCOUNT_BAD_ACTION = -0.1
GOAL_REWARD = 10.
HOLES_REWARD = -10.


class EnvObj(object):
    def __init__(self, coords, reward, name):
        # self.observation_space = spaces.Box(0, 255, [240, 320], dtype=np.float32)
        #
        # self.game_path = game_path
        # self.is_visible = is_visible
        # self.env = self.create_enviroment(game_path, is_visible)
        # self.seed(1)

        self.x = coords[0]
        self.y = coords[1]
        self.reward = reward
        self.name = name


class Maze(Env):
    def __init__(self, game_path, height=4, width=4, num_actions=4, is_visible=True):
        self.maze_h = height
        self.maze_w = width
        # self.action_space = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] #['u', 'd', 'l', 'r']
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(0, height, [width, height], dtype=np.float32)
        self.is_visible = is_visible
        self.game_path = game_path

        self.env = self.create_enviroment(game_path, is_visible)

        # self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = num_actions


        self.obj = {}
        self.done = False

    def create_enviroment(self, game_path, is_visible):
        env = vz.DoomGame()
        env.load_config("{}.cfg".format(game_path))
        env.set_doom_scenario_path("{}.wad".format(game_path))
        env.set_window_visible(is_visible)
        env.init()
        self.is_init = True
        return env


    def init(self, random):
        '''
        Bluind the state space
        :return:
        '''

        for idx, x_coord in enumerate(range(1, self.maze_w - 1)):
            self.obj["hell{}".format(idx)] = EnvObj((x_coord, 0), HOLES_REWARD, "hell{}".format(idx))


        if random:
            sample = np.random.random()
            if sample > 0.5:
                self.obj["goal"] = EnvObj((self.maze_w-1, 0), GOAL_REWARD, "goal")
            else:
                self.obj["goal"] = EnvObj((self.maze_w - 1, self.maze_h-1), GOAL_REWARD, "goal")
        else:
            self.obj["goal"] = EnvObj((self.maze_w - 1, 0), GOAL_REWARD, "goal")


    def new_episode(self):
        """
        Create a new starting point
        :return:
        """
        self.obj["hero"] = EnvObj((0, 0), 0, "hero")
        self.block_position = []
        self.done = False
        return [0, 0]


    def make_action(self, action):
        """
        ALMOST an ALIAS for step
        :param action:
        :return:
        """
        s_, discount_action_reward = self.move_hero(action)  # move the hero
        reward, done = self.check_goal(s_)
        reward += discount_action_reward
        self.done = done
        return reward


    def is_episode_finished(self):
        return self.done

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

        assert action < self.n_actions

        # take action
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

    def render(self, step=None):
        if step:
            img = np.ones([(self.maze_h * UNIT) + 40, self.maze_w * UNIT, 3]) * 255.0
        else:
            img = np.ones([(self.maze_h * UNIT), self.maze_w * UNIT, 3]) * 255.0
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
                     fill=(152, 152, 204))

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

        if step:
            font = ImageFont.truetype("./image/FreeSans.ttf", 18)
            draw.text((140, 170), 'Step: ' + str(step), (0, 0, 0), font=font)

        return img

    def render_np(self):
        img = self.render()
        return np.array(img)

    def render_torch(self, img_tran=T.ToTensor(), step=None):
        img = self.render(step)
        img = img_tran(img)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.show()

        return img


if __name__ == "__main__":
    env = Maze(height=4, width=8)
    img = env.render_torch()
    print(img)
    # img = env.render()
    # img.save("env.png", format="png")
