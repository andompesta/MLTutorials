import argparse
import random                # Handling random number generation
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from visdom import Visdom
import RL.helper as helper
import itertools
from functools import reduce
import matplotlib.pyplot as plt
from collections import deque

import gym
from RL.wrappers import Monitor, ToTorchObs
from os import path
from RL.PPO.model_ppo import PPO
from RL.PPO.memory_collector import MemoryCollector

EXP_NAME = "exp-ppo-{}".format(datetime.now().strftime("%H:%M:%S"))
GLOBAL_STEP = 0
P_LOSSES = []
RESET_FRAME = torch.zeros([1, 4, 34, 136])
HISTOGRAM = {0:0, 1:0, 2:0, 3:0}


def __pars_args__():
    parser = argparse.ArgumentParser(description='PPO')

    parser.add_argument('-gam', '--gamma', type=float, default=0.95, help='discounting factor')
    parser.add_argument('-lam', '--lam', type=float, default=0.99, help='discounting factor')
    parser.add_argument('-m_grad', '--max_grad_norm', type=float, default=0.5, help='discounting factor')
    parser.add_argument('-ent_coef', '--ent_coef', type=float, default=0.,
                        help='policy entropy coefficient in the optimization objective')
    parser.add_argument('-vf_coef', '--vf_coef', type=float, default=0.5,
                        help='value function loss coefficient in the optimization objective')
    parser.add_argument("-tot_t", "--total_timesteps", type=int, default=10000000,
                        help="number of timesteps (i.e. number of actions taken in the environment)")

    parser.add_argument('-n_train_ep', '--number_train_epoch', type=int, default=4,
                        help='number of training epochs per update')
    parser.add_argument('-clp_rng', '--clip_range', type=float, default=0.2,
                        help='clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the ' +
                             'training and 0 is the end of the training')


    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size used during learning')
    parser.add_argument('-mini_bs', '--mini_batchs', type=int, default=4, help='number of training minibatches per update. For recurrent policies, should be smaller or equal than number of environments run in parallel.')
    parser.add_argument('-n_step', '--n_step', type=int, default=1024, help='number of steps of the vectorized environment per update')

    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./monitor', help='Path to save monitor of agent')
    parser.add_argument('-v', '--version', default='0', help='Path to save monitor of agent')

    parser.add_argument('-save_every', '--save_every', type=int, default=0,
                        help='number of timesteps between saving events')

    parser.add_argument('-log_every', '--log_every', type=int, default=10,
                        help='number of timesteps between logs events')


    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[56, 84], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=False, help="Use cuda")

    return parser.parse_args()




def build_env(args, env_name=EXP_NAME):
    env = gym.make("CartPole-v0")
    # env = ToTorchObs(env)
    env = Monitor(env, helper.ensure_dir(path.join(args.monitor_path, env_name)), allow_early_resets=True)
    return env

def test_environment(args, trs_img):
    env = build_env(args)

    episodes = 10
    for i in range(episodes):
        obs = env.reset()
        for t in range(100):
            img = env.render(mode='rgb_array')
            print(obs)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    img = trs_img(img)
    plt.figure()
    plt.imshow(img[0, :, :].numpy(), cmap="gray")
    plt.show()
    plt.close()
    env.close()

def step_setup(args, train_model, device):
    optimizer = torch.optim.Adam(train_model.parameters(), lr=args.learning_rate)

    def train_step_fn(obs, returns, old_actions, old_values, old_neg_log_prbs):
        obs = torch.tensor(obs).float().to(device)
        returns = torch.tensor(returns).float().to(device)
        old_values = torch.tensor(old_values).float().to(device)
        old_neg_log_prbs = torch.tensor(old_neg_log_prbs).float().to(device)
        old_actions = torch.tensor(old_actions).to(device)

        with torch.set_grad_enabled(False):
            advantages = returns - old_values
            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.set_grad_enabled(True):
            train_model.train()
            train_model.zero_grad()

            value_f, actions, neg_log_probs, entropy = train_model(obs, action=old_actions)

            assert(actions.sum().item() == old_actions.sum().item())

            loss, pg_loss, value_loss, approx_kl, clip_frac = train_model.loss(returns, value_f, neg_log_probs, entropy, advantages,
                                                                               old_values, old_neg_log_prbs,
                                                                               args.clip_range, args.ent_coef, args.vf_coef)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
            optimizer.step()

        return list(map(lambda x: x.detach().item(), [loss, pg_loss, value_loss, approx_kl, clip_frac]))

    return train_step_fn, optimizer





if __name__ == '__main__':
    vis = Visdom()
    args = __pars_args__()
    img_trs = T.Compose([T.ToPILImage(),
                           T.Resize(args.state_size),
                           T.Grayscale(),
                           T.ToTensor()])

    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    # test_environment(args, img_trs)
    env = build_env(args)

    # model param
    obs_size = reduce((lambda x, y: x * y), env.observation_space.shape)
    action_space = env.action_space.n
    # used for sampling
    model = PPO(obs_size, args.hidden_dim, action_space,
                n_step=args.n_step)
    model.to(device)

    train_fn, optm = step_setup(args, model, device)

    memory_collector = MemoryCollector(env, model, args.n_step, args.gamma, args.lam, device)
    ep_info_buf = deque(maxlen=100)

    n_env = 1
    n_batch = n_env * args.n_step
    n_updates = args.total_timesteps // args.batch_size
    n_batch_train = n_batch // args.mini_batchs

    for update in range(1, n_updates+1):
        assert n_batch % args.mini_batchs == 0

        # Start timer
        frac = 1.0 - (update - 1.0) / n_updates

        if update % args.log_every == 0:
            print('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neg_log_prb, ep_infos = memory_collector.run()

        if update % args.log_every == 0:
            print('Done.')

        ep_info_buf.extend(ep_infos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mb_loss_vals = []

        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(n_batch)
        for _ in range(args.number_train_epoch):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, n_batch, n_batch_train):
                end = start + n_batch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neg_log_prb))
                loss, pg_loss, value_loss, approx_kl, clip_frac = train_fn(*slices)
                mb_loss_vals.append(loss)


        # Feedforward --> get losses --> update
        loss_vals = np.mean(mb_loss_vals, axis=0)
        loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        if update % args.log_every == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = helper.explained_variance(values, returns)
            print("misc/serial_timesteps", update * args.n_steps)
            print("misc/n_updates", update)
            print("misc/total_timesteps", update * n_batch)

            print("misc/explained_variance", float(ev))
            print('ep_rew_mean', helper.safemean([ep_info['r'] for ep_info in ep_info_buf]))
            print('ep_len_mean', helper.safemean([ep_info['l'] for ep_info in ep_info_buf]))
            for (loss_val, loss_name) in zip(loss_vals, loss_names):
                print('loss/' + loss_name, loss_val)

        if update % args.save_every == 0 or update == 1:
            helper.save_checkpoint({
                'update': update,
                'state_dict': model.state_dict(),
                'optimizer': optm.state_dict()
            },
                path=args.model_path,
                filename='train_net.cptk',
                version=args.version
            )

    env.close()
