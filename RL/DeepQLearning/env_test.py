import gym
import numpy as np
import RL.DeepQLearning.helper as helper
import matplotlib.pyplot as plt
from torchvision import transforms

env = gym.make("Breakout-v0")

print("Action space size: {}".format(env.action_space.n))
print(env.unwrapped.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

img_array = env.render(mode='rgb_array')




plt.figure()
plt.imshow(img_array)

#test
img_trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((84, 84)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])

img_resize = helper.state_processor(img_array, (34, 0, 0, 0), img_trans)
plt.figure()
plt.imshow(img_resize[:, :, 0].numpy(), cmap="gray")
plt.show()

#test batch
# img_resize = helper.img_resize(img_croped, [1, 0.5, 0.5, 1])

# plt.figure()
# plt.imshow(img_array)
# plt.figure()
# plt.imshow(img_grey[0, :, :, 0], cmap="gray")
# plt.figure()
# plt.imshow(img_croped[0, :, :, 0], cmap="gray")
# plt.figure()
# plt.imshow(img_resize[0, :, :, 0], cmap="gray")


# [env.step(2) for x in range(1)]
# plt.figure()
# plt.imshow(env.render(mode='rgb_array'))
#
# env.render(close=True)
#
# plt.imshow(observation[34:-16,:,:])
