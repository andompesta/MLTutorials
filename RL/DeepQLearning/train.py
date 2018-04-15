import itertools
import sys
from os.path import join as path_join
import numpy as np
import torch
from torch.autograd import Variable
from RL.DeepQLearning.model_dqn import epsilon_greedy_policy
import RL.DeepQLearning.helper as helper
from collections import deque
import random


if "../" not in sys.path:
  sys.path.append("../")

GLOBAL_STEP = 0

def work(env, q_network, t_network, args, vis, exp_name, optimizer):
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
    video_path = path_join(args.monitor_path, exp_name)
    summary_path = path_join(args.model_path, exp_name)
    helper.ensure_dir(summary_path)
    helper.ensure_dir(video_path)
    stack_frame = helper.stack_frame_setup(args.state_size)

    if optimizer is None:
        optimizer = torch.optim.Adagrad(q_network.parameters(), lr=args.learning_rate)

    # Keeps track of useful statistics
    replay_memory = helper.ExperienceBuffer(buffer_size=args.replay_memory_size)


    # The policy we're following
    policy = epsilon_greedy_policy(q_network, args.epsilon_end, args.epsilon_start, args.epsilon_decay_rate, args.actions)
    stacked_frames = deque([torch.zeros(args.state_size) for i in range(args.number_frames)], maxlen=4)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    env.new_episode()
    frame = env.get_state().screen_buffer
    state = stack_frame(stacked_frames, frame)
    for i in range(args.replay_memory_init_size):
        action = random.choice(args.actions)
        reward = env.make_action(action)
        done = env.is_episode_finished()

        if done:
            stacked_frames.append(torch.zeros(args.state_size))
            replay_memory.add(helper.Transition(state, action, reward, RESET_FRAME, float(done)))
            env.new_episode()
        else:
            next_frame = env.get_state().screen_buffer
            next_state = stack_frame(stacked_frames, next_frame)
            replay_memory.add(helper.Transition(state, action, reward, next_state, float(done)))
            state = next_state

    stacked_frames.append(torch.zeros(args.state_size)) # add one empty frame to indicate a new episode
    for i_episode in range(args.num_episodes):
        # Save the current checkpoint
        torch.save(q_network.state_dict(), helper.ensure_dir(
            path_join(summary_path, 'q_net-{}.cptk'.format(i_episode))))
        torch.save(t_network.state_dict(), helper.ensure_dir(
            path_join(summary_path, 't_net-{}.cptk'.format(i_episode))))

        env.new_episode()
        frame = env.get_state().screen_buffer
        state = stack_frame(stacked_frames, frame)


        # One step in the environment
        for t in itertools.count():
            # Update params
            if GLOBAL_STEP % args.update_target_estimator_every == 0:
                t_network.load_state_dict(q_network.state_dict())
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}".format(t, GLOBAL_STEP, i_episode + 1), end="")
            sys.stdout.flush()


            action, epsilon = policy(state, GLOBAL_STEP)
            reward = env.make_action(action)
            done = env.is_episode_finished()

            if done:
                stacked_frames.append(torch.zeros(args.state_size))
                replay_memory.add(helper.Transition(state, action, reward, RESET_FRAME, float(done)))
            else:
                next_frame = env.get_state().screen_buffer
                next_state = stack_frame(stacked_frames, next_frame)
                replay_memory.add(helper.Transition(state, action, reward, next_state, float(done)))
                state = next_state


            # OPTIMIZE MODEL
            # Sample a minibatch from the replay memory
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_memory.sample(args.batch_size)
            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = torch.ByteTensor(tuple(map(lambda s: not torch.equal(s, RESET_FRAME), next_state_batch)))
            non_final_next_states = Variable(torch.stack([s for s in next_state_batch if not torch.equal(s, RESET_FRAME)]), volatile=True)
            estimated_next_q_values = Variable(torch.zeros(args.batch_size))

            if helper.use_cuda:
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                non_final_mask = non_final_mask.cuda()
                non_final_next_states = non_final_next_states.cuda()
                estimated_next_q_values = estimated_next_q_values.cuda()

            # Calculate q values and targets
            estimated_q_value = q_network.compute_q_value(Variable(state_batch), Variable(action_batch))


            assert t_network.training == False, "target network is training"
            # remove reward of final state
            estimated_next_q_values[non_final_mask] = t_network.forward(non_final_next_states).max(dim=1)[0]

            expected_q_values = reward_batch + (args.discount_factor * estimated_next_q_values.data)
            expected_q_values = Variable(expected_q_values)

            # Perform gradient descent update
            optimizer.zero_grad()
            loss = q_network.compute_loss(estimated_q_value, expected_q_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

            p_losses.append([loss.data[0], estimated_q_value.max().data[0], epsilon])
            GLOBAL_STEP += 1

            vis.line(Y=np.array(p_losses),
                     X=np.repeat(np.expand_dims(np.arange(GLOBAL_STEP), 1), 3, axis=1),
                     opts=dict(legend=["loss", "max_q_value", "epsilon"],
                               title="q_network",
                               showlegend=True),
                     win="plane_q_network_{}".format(exp_name))


            if (t == args.max_steps) or done:
                stat = helper.EpisodeStat(t, env.get_total_reward())

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