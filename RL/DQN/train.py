import itertools
import sys
from os import path
import numpy as np
import torch
from RL.DQN.model_dqn import epsilon_greedy_policy
import RL.helper as helper
from collections import deque
import random


if "../" not in sys.path:
  sys.path.append("../")

GLOBAL_STEP = 0

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
    RESET_FRAME = torch.zeros([4] + args.state_size)

    torch.manual_seed(args.seed)

    # local variable for plotting
    p_rewards = []
    p_losses = []

    # path variable
    video_path = path.join(args.monitor_path, exp_name)
    summary_path = path.join(args.model_path, exp_name)
    helper.ensure_dir(summary_path)
    helper.ensure_dir(video_path)

    # stack_frame_fn = helper.stack_frame_setup(args.state_size,
    #                                           top_offset_height=50,
    #                                           bottom_offset_height=10,
    #                                           left_offset_width=30,
    #                                           right_offset_width=30
    #                                           )

    stack_frame_fn = helper.stack_frame_setup(args.state_size,
                                              top_offset_height=0,
                                              bottom_offset_height=0,
                                              left_offset_width=0,
                                              right_offset_width=0
                                              )

    if optimizer is None:
        optimizer = torch.optim.Adagrad(q_network.parameters(), lr=args.learning_rate)

    # Keeps track of useful statistics
    replay_memory = helper.ExperienceBuffer(buffer_size=args.replay_memory_size)


    # The policy we're following
    policy = epsilon_greedy_policy(q_network,
                                   args.epsilon_end,
                                   args.epsilon_start,
                                   args.epsilon_decay_rate,
                                   args.actions, device)

    stacked_frames = deque([torch.zeros(args.state_size) for i in range(args.number_frames)], maxlen=args.number_frames)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")

    # env.new_episode()
    # frame = env.get_state().screen_buffer
    env.reset()
    frame = helper.get_screen(env)

    state = stack_frame_fn(stacked_frames, frame, True)
    for i in range(args.replay_memory_init_size):
        # action = random.choice(args.actions)
        # reward = env.make_action(action)
        # done = env.is_episode_finished()

        action_idx = env.action_space.sample()
        action = args.actions[action_idx]
        _, reward, done, _ = env.step(action_idx)

        if done:
            transition = helper.Transition(state, action, reward, RESET_FRAME, int(done))
            replay_memory.store(transition)

            # start new episode
            # env.new_episode()
            # frame = env.get_state().screen_buffer
            env.reset()
            frame = helper.get_screen(env)

            state = stack_frame_fn(stacked_frames, frame, True)
        else:
            # frame = env.get_state().screen_buffer
            frame = helper.get_screen(env)

            next_state = stack_frame_fn(stacked_frames, frame)
            transition = helper.Transition(state, action, reward, next_state, int(done))
            replay_memory.store(transition)
            state = next_state

    update_param = True  # variable use to update the parameter of the t_net

    for i_episode in range(args.num_episodes):

        if update_param:
            # update t_net params
            update_param = False
            t_network.load_state_dict(q_network.state_dict())
            print("\nCopied model parameters to target network.")

        # Save the current checkpoint
        if i_episode % 200 == 0:
            helper.save_checkpoint({
                'episode': i_episode,
                'state_dict': q_network.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': GLOBAL_STEP,
            },
                path=summary_path,
                filename='q_net-{}.cptk'.format(i_episode),
                version=args.version
            )

            helper.save_checkpoint({
                'episode': i_episode,
                'state_dict': t_network.cpu().state_dict(),
            },
                path=summary_path,
                filename='t_net-{}.cptk'.format(i_episode),
                version=args.version
            )


        # env.new_episode()
        # frame = env.get_state().screen_buffer
        # state = stack_frame_fn(stacked_frames, frame, True)

        env.reset()
        frame = helper.get_screen(env)
        state = stack_frame_fn(stacked_frames, frame, True)
        total_reward = 0



        # One step in the environment
        for t in itertools.count():
            # Update params
            if GLOBAL_STEP % args.update_target_estimator_every == 0:
                update_param = True

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}".format(t, GLOBAL_STEP, i_episode + 1), end="")
            sys.stdout.flush()

            action_idx, epsilon = policy(state, GLOBAL_STEP)
            action = args.actions[action_idx]

            _, reward, done, _ = env.step(action_idx)
            total_reward += reward

            # action, epsilon = policy(state, GLOBAL_STEP)
            # reward = env.make_action(action)
            # done = env.is_episode_finished()

            if done:
                transition = helper.Transition(state, action, reward, RESET_FRAME, int(done))
                replay_memory.store(transition)

                # start new episode
                # env.new_episode()
                # frame = env.get_state().screen_buffer
                env.reset()
                frame = helper.get_screen(env)
                state = stack_frame_fn(stacked_frames, frame, True)
            else:
                # frame = env.get_state().screen_buffer
                frame = helper.get_screen(env)
                next_state = stack_frame_fn(stacked_frames, frame)
                transition = helper.Transition(state, action, reward, next_state, int(done))

                replay_memory.store(transition)
                state = next_state

            # OPTIMIZE MODEL
            # Sample a minibatch from the replay memory
            batch = replay_memory.sample(args.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.stack(state_batch, dim=0).to(device)
            action_batch = torch.tensor(action_batch).float().to(device)
            reward_batch = torch.tensor(reward_batch).to(device)
            next_state_batch = torch.stack(next_state_batch, dim=0).to(device)
            done_batch = torch.tensor(done_batch).to(device)


            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = (1 - done_batch).byte().to(device)
            non_final_next_states = next_state_batch[non_final_mask].to(device)
            q_target_next_state = torch.zeros(args.batch_size).to(device)

            assert t_network.training == False, "target network is training"
            with torch.no_grad():
                # remove reward of final state
                q_target_next_state[non_final_mask] = t_network.forward(non_final_next_states).max(dim=1)[0]
                q_target_batch = reward_batch + (args.discount_factor * q_target_next_state)

            # Calculate q values and targets
            with torch.set_grad_enabled(True):
                q_value_batch = q_network.compute_q_value(state_batch, action_batch)
                loss = q_network.compute_loss(q_value_batch, q_target_batch)

            # Perform gradient descent update
            q_network.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(q_network.parameters(), args.max_grad)
            optimizer.step()

            p_losses.append([loss.item(), q_value_batch.max().item(), epsilon])
            GLOBAL_STEP += 1

            vis.line(Y=np.array(p_losses),
                     X=np.repeat(np.expand_dims(np.arange(GLOBAL_STEP), 1), 3, axis=1),
                     opts=dict(legend=["loss", "max_q_value", "epsilon"],
                               title="q_network",
                               showlegend=True),
                     win="plane_q_network_{}".format(exp_name))


            if (t == args.max_steps) or done:
                stat = helper.EpisodeStat(t, total_reward)

                print('Episode: {}'.format(i_episode),
                      'Episode length: {}'.format(stat.episode_length),
                      'Episode reward: {}'.format(stat.episode_reward),
                      'Avg reward: {}'.format(stat.avg_reward),
                      'Explore P: {:.4f}'.format(epsilon))
                break


        p_rewards.append([stat.episode_reward, stat.episode_length, stat.avg_reward])
        vis.line(Y=np.array(p_rewards),
                 X=np.repeat(np.expand_dims(np.arange(i_episode + 1), 1), 3, axis=1),
                 opts=dict(legend=["episode_reward", "episode_length", "average_reward"],
                           title="rewards",
                           showlegend=True),
                 win="plane_reward_{}".format(exp_name))

        # vis.save(["main"])
        yield GLOBAL_STEP, stat

    return stat