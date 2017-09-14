import gym
import numpy as np
from matplotlib import pyplot as plt
import DeepQLearning.helper as helper

env = gym.make("Breakout-v0")

print("Action space size: {}".format(env.action_space.n))
print(env.unwrapped.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))


img_array = env.render(mode='rgb_array')
img_grey = helper.img_rgb2gray(np.tile(img_array, [2, 1, 1, 1]))
img_croped = helper.img_crop_to_bounding_box(img_grey, 34, 0, 160, 160)
img_resize = helper.img_resize(img_croped, [1, 0.5, 0.5, 1])

plt.figure()
plt.imshow(img_array)
plt.figure()
plt.imshow(img_grey[0, :, :, 0], cmap="gray")
plt.figure()
plt.imshow(img_croped[0, :, :, 0], cmap="gray")
plt.figure()
plt.imshow(img_resize[0, :, :, 0], cmap="gray")


[env.step(2) for x in range(1)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)

plt.imshow(observation[34:-16,:,:])
