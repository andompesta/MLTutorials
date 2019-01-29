"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels


class Maze(tk.Tk, object):
    def __init__(self, title, height=4, width=4):
        super(Maze, self).__init__()
        self.maze_h = height
        self.maze_w = width
        
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title(title)
        self.geometry('{0}x{1}'.format(self.maze_w * UNIT, self.maze_h * UNIT))
        self._build_maze()

    def _build_maze(self):
        '''
        Bluind the state space
        :return: 
        '''
        self.canvas = tk.Canvas(self, bg='white', height=self.maze_h * UNIT, width=self.maze_w * UNIT)

        # create grids
        for c in range(0, self.maze_w * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.maze_h * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.maze_h * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.maze_w * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        self.hells = []
        for x_coord in range(1, self.maze_w - 1):
            hell_center = origin + np.array([UNIT * x_coord, 0])
            self.hells.append(
                self.canvas.create_rectangle(hell_center[0] - 15, hell_center[1] - 15,
                                             hell_center[0] + 15, hell_center[1] + 15,
                                             fill='black'))
        self.hells_coords = [self.canvas.coords(hell) for hell in self.hells]

        # # hell
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 15, hell1_center[1] - 15,
        #     hell1_center[0] + 15, hell1_center[1] + 15,
        #     fill='black')
        #
        # # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin + np.array([UNIT * (self.maze_w - 1), 0])
        self.oval = self.canvas.create_oval(oval_center[0] - 15, oval_center[1] - 15,
                                            oval_center[0] + 15, oval_center[1] + 15,
                                            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                 origin[0] + 15, origin[1] + 15,
                                                 fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        """
        Create a new starting point
        :return: 
        """
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        """
        Execute the given action
        :param action: 
        :return: 
        """
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (self.maze_h - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (self.maze_w - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ in self.hells_coords:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()