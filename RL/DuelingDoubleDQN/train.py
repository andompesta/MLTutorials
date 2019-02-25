import itertools
import sys
from os import path
import numpy as np
import torch
from RL.DuelingDoubleDQN.model_dd_dqn import epsilon_greedy_policy
import RL.helper as helper
from collections import deque
import random
from torchvision import transforms as T



if "../" not in sys.path:
  sys.path.append("../")

GLOBAL_STEP = 0
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 100000



def work(env, q_network, t_network, args, vis, exp_name, optimizer, device):
    """
    Train the model
    :param env: OpenAI environment
    :param q_network: q-valueRL. network 
    :param t_network: target network
    :param args: args to be used
    :param vis: Visdom server
    :param exp_name: experiment name
    :param optimizer: optimizer used 
    :return: 
    """
    global GLOBAL_STEP
    global PRIORITY_BETA_FRAMES
    global PRIORITY_BETA_START
    global PRIORITY_ALPHA

    IMG_TRANS = T.Compose([T.ToPILImage(),
                           T.CenterCrop([170, 280]),
                           T.Resize(args.state_size),
                           T.Grayscale(),
                           T.ToTensor()])

    ACTION_MAP = np.identity(len(args.actions), dtype=int).tolist()
    RESET_FRAME = torch.zeros([1, args.number_frames, args.state_size[0], args.state_size[1]])

    torch.manual_seed(args.seed)

    # local variable for plotting
    p_rewards = []
    p_losses = []

    # path variable
    video_path = path.join(args.monitor_path, exp_name)
    summary_path = path.join(args.model_path, exp_name)
    helper.ensure_dir(summary_path)
    helper.ensure_dir(video_path)


    stack_frame_fn = helper.stack_frame_setup(IMG_TRANS)

    # Keeps track of useful statistics
    # replay_memory = helper.ExperienceBuffer(args.replay_memory_size)
    replay_memory = helper.PrioritizedReplayMemory(capacity=args.replay_memory_size,
                                                   alpha=PRIORITY_ALPHA,
                                                   beta_start=PRIORITY_BETA_START,
                                                   beta_frames=PRIORITY_BETA_FRAMES)


    # The policy we're following
    policy = epsilon_greedy_policy(q_network,
                                   args.epsilon_end,
                                   args.epsilon_start,
                                   args.epsilon_decay_rate,
                                   args.actions,
                                   device)

    frames_queue = deque(maxlen=args.number_frames)

    print("Populating replay memory...")
    frame = env.reset()

    state = stack_frame_fn(frames_queue, frame, True)
    for i in range(args.replay_memory_init_size):

        action = random.choice(args.actions)
        frame, reward, done, info = env.step(ACTION_MAP[action])

        if done:
            transition = helper.Transition(state, action, RESET_FRAME, reward, int(done))
            replay_memory.store(transition)

            # start new episode
            frame = env.reset()
            state = stack_frame_fn(frames_queue, frame, True)
        else:
            next_state = stack_frame_fn(frames_queue, frame)

            transition = helper.Transition(state, action, next_state, reward, int(done))
            replay_memory.store(transition)
            state = next_state

    print("start training")
    for i_episode in range(args.num_episodes):

        if i_episode % 200 == 0:
            # Save the current checkpoint
            helper.save_checkpoint({
                'episode': i_episode,
                'state_dict': q_network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': GLOBAL_STEP,
            },
                path=summary_path,
                # filename='q_net-{}.cptk'.format(i_episode)
                filename='q_net.cptk',
                version=args.version
            )

            helper.save_checkpoint({
                'episode': i_episode,
                'state_dict': t_network.state_dict(),
            },
                path=summary_path,
                # filename='t_net-{}.cptk'.format(i_episode),
                filename='t_net.cptk',
                version=args.version
            )

        frame = env.reset()
        total_reward = 0

        state = stack_frame_fn(frames_queue, frame, True)
        # One step in the environment
        for t in itertools.count():
            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}".format(t, GLOBAL_STEP, i_episode + 1), end="")
            sys.stdout.flush()

            action, epsilon = policy(state, GLOBAL_STEP)
            frame, reward, done, info = env.step(ACTION_MAP[action])
            total_reward += reward

            if done:
                transition = helper.Transition(state, action, RESET_FRAME, reward, int(done))
                replay_memory.store(transition)
            else:
                next_state = stack_frame_fn(frames_queue, frame)
                transition = helper.Transition(state, action, next_state, reward, int(done))
                replay_memory.store(transition)
                state = next_state


            # OPTIMIZE MODEL
            # Sample a minibatch from the replay memory
            batch, idx, weights = replay_memory.sample(args.batch_size)
            # batch = replay_memory.sample(args.batch_size)
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)

            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(1)
            reward_batch = torch.tensor(reward_batch).to(device)
            done_batch = torch.tensor(done_batch).to(device)
            next_state_batch = torch.cat(next_state_batch).to(device)
            weights = torch.tensor(weights, dtype=torch.float32).float().to(device)
            # weights = torch.ones((args.batch_size), device=device)

            assert t_network.training == False, "target network is training"
            with torch.no_grad():
                non_final_mask = (1 - done_batch).byte()
                non_final_next_states = next_state_batch[non_final_mask]
                next_q_value_batch = torch.zeros(args.batch_size).to(device)

                next_q_value_batch[non_final_mask] = t_network.forward(non_final_next_states).max(dim=1)[0]
                next_q_value_batch = reward_batch + (args.discount_factor * next_q_value_batch)


            # Calculate q values and targets
            with torch.set_grad_enabled(True):
                q_value_batch = q_network.compute_q_value(state_batch, action_batch)

            # Perform gradient descent update
            loss, td_error, diff = q_network.compute_loss(q_value_batch, next_q_value_batch, weights)

            q_network.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad)
            optimizer.step()

            assert next_q_value_batch.grad_fn == None

            # Update priority
            replay_memory.update_priorities(idx, diff.detach().squeeze().cpu().numpy().tolist())

            p_losses.append([loss.item(), td_error.item(), q_value_batch.max().item(), next_q_value_batch.max().item(), epsilon])
            GLOBAL_STEP += 1

            vis.line(Y=np.array(p_losses),
                     X=np.repeat(np.expand_dims(np.arange(GLOBAL_STEP), 1), 5, axis=1),
                     opts=dict(legend=["loss", "td_error", "max_q", "max_n_q", "epsilon"],
                               title="q_network",
                               showlegend=True),
                     win="plane_q_network_{}_{}".format(exp_name, args.version))

            if (t == args.max_steps) or done:
                stat = helper.EpisodeStat(t, total_reward)

                print('Episode: {}'.format(i_episode),
                      'Episode length: {}'.format(stat.episode_length),
                      'Episode reward: {}'.format(stat.episode_reward),
                      'Avg reward: {}'.format(stat.avg_reward),
                      'Avg length: {}'.format(stat.avg_length),
                      'Explore P: {:.4f}'.format(epsilon))
                break

        # Update params
        if i_episode % args.update_target_estimator_every == 0:
            t_network.load_state_dict(q_network.state_dict())
            print("\nCopied model parameters to target network.")

        p_rewards.append([stat.episode_reward, stat.episode_length, stat.avg_reward])
        vis.line(Y=np.array(p_rewards),
                 X=np.repeat(np.expand_dims(np.arange(i_episode + 1), 1), 3, axis=1),
                 opts=dict(legend=["episode_reward", "episode_length", "average_reward"],
                           title="rewards",
                           showlegend=True),
                 win="plane_reward_{}_".format(exp_name, args.version))

        # vis.save(["main"])
        yield GLOBAL_STEP, stat

    return stat