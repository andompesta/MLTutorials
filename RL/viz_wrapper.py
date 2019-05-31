import numpy as np
import vizdoom as vz
from os import path, kill
import gym
from gym import error, spaces, utils


class VizdoomEnv(gym.Env):
    def __init__(self, game_path, is_visible, num_actions):
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(0, 255, [240, 320], dtype=np.float32)

        self.game_path = game_path
        self.is_visible = is_visible
        self.env = self.create_enviroment(game_path, is_visible)
        self.seed(1)


    def create_enviroment(self, game_path, is_visible):
        env = vz.DoomGame()
        env.load_config("{}.cfg".format(game_path))
        env.set_doom_scenario_path("{}.wad".format(game_path))
        env.set_window_visible(is_visible)
        env.init()
        self.is_init = True
        return env

    def seed(self, seed=None):
        if isinstance(seed, int):
            self.env.set_seed(seed)

    def step(self, action):
        reward = self.env.make_action(action)

        if self.env.is_player_dead() or self.env.is_episode_finished():
            done = True
            frame = None
        else:
            done = False
            try:
                frame = self.env.get_state().screen_buffer
            except AttributeError:
                import time
                import subprocess, signal
                time.sleep(2)

                p = subprocess.Popen(['ps', 'ax'], stdout=subprocess.PIPE)
                out, err = p.communicate()

                for line in out.splitlines():
                    if 'vizdoom' in str(line):
                        pid = int(line.split(None, 1)[0])
                        kill(pid, signal.SIGKILL)
                        print("vizdoom killed")

                time.sleep(2)
                print("step hard restart")
                self.env = self.create_enviroment(self.game_path, self.is_visible)
                reward = 0
                done = False
                frame = self.reset()

        info = vars(self).copy()
        info.pop('env', None)  # infos for openai baselines need to be picklable, game is not
        return frame, reward, done, info

    def reset(self):
        self.env.new_episode()
        try:
            frame = self.env.get_state().screen_buffer
        except AttributeError:
            import time
            import subprocess, signal
            time.sleep(2)

            p = subprocess.Popen(['ps', 'ax'], stdout=subprocess.PIPE)
            out, err = p.communicate()

            for line in out.splitlines():
                if 'vizdoom' in str(line):
                    pid = int(line.split(None, 1)[0])
                    kill(pid, signal.SIGKILL)
                    print("vizdoom killed")

            time.sleep(2)
            print("reset hard restart")
            self.env = self.create_enviroment(self.game_path, self.is_visible)
            frame = self.reset()

        return frame

    def render(self, mode='rgb_array'):
        raise NotImplementedError('Rendering is not implemented!')


