import gym
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

env = gym.make("CartPole-v0")

print("Action space size: {}".format(env.action_space.n))
observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

img_array = env.render(mode='rgb_array')


observation = env.reset()
for _ in range(1000):
  img_array = env.render(mode='rgb_array')
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()



plt.figure()
plt.imshow(img_array)

#test
img_trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((56, 84)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])
img_proc = img_trans(img_array)

plt.figure()
plt.imshow(img_proc[0, :, :].numpy(), cmap="gray")
plt.show()

# plt.figure()
# plt.imshow(img_array)
# plt.figure()
# plt.imshow(img_grey[0, :, :, 0], cmap="gray")
# plt.figure()
# plt.imshow(img_croped[0, :, :, 0], cmap="gray")
# plt.figure()
# plt.imshow(img_resize[0, :, :, 0], cmap="gray")

plt.close()
env.close()
