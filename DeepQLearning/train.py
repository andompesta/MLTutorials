import torch
import torch.optim as optim
import numpy as np
from gym.wrappers import Monitor
import DeepQLearning.helper as helper
from DeepQLearning.model_dqn import epsilon_greedy_policy
from os.path import join as path_join

import itertools
import sys
import psutil
if "../" not in sys.path:
  sys.path.append("../")

import DeepQLearning.plotting as plotting
from collections import deque, namedtuple

def work(env, q_network, t_network, args, summary, summary_path, video_path, optimizer):
    """
    Train the model
    :param env: OpenAI environment
    :param q_network: q-value network 
    :param t_network: target network
    :param args: args to be used
    :param summary: crayon experiment
    :param summary_path: experiment path
    :param video_path: video path
    :param optimizer: optimizer used 
    :return: 
    """
    torch.manual_seed(args.seed)
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    if optimizer is None:
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(args.num_episodes),
        episode_rewards=np.zeros(args.num_episodes))
    replay_memory = helper.ExperienceBuffer()

    # The epsilon decay schedule
    epsilons = np.linspace(args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)

    # The policy we're following
    policy = epsilon_greedy_policy(q_network)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state = helper.state_processor(state)
    state = np.stack([state] * 4, axis=0)
    for i in range(args.replay_memory_init_size):
        action_probs = policy(state, epsilons[min(t_network.step, args.epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)     # initially take random action
        next_state, reward, done, _ = env.step(args.actions[action])
        next_state = helper.state_processor(next_state)
        next_state = np.append(state[1:, :, :], np.expand_dims(next_state, 0), axis=0)
        replay_memory.add(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = helper.state_processor(state)
            state = np.stack([state] * 4, axis=0)
        else:
            state = next_state

    # Record videos
    env = Monitor(env, directory=video_path, video_callable=lambda count: count % args.record_video_every == 0,
                  resume=True)

    for i_episode in range(args.num_episodes):
        # Save the current checkpoint
        torch.save(q_network.state_dict(), helper.ensure_dir(
            path_join(summary_path, 'q_net-{}.cptk'.format(i_episode))))
        torch.save(t_network.state_dict(), helper.ensure_dir(
            path_join(summary_path, 't_net-{}.cptk'.format(i_episode))))

        # Reset the environment
        state = env.reset()
        state = helper.state_processor(state)
        state = np.stack([state] * 4, axis=0)
        loss = None

        # One step in the environment
        for t in itertools.count():
            # Epsilon for this time step
            epsilon = epsilons[min(t_network.step, args.epsilon_decay_steps - 1)]

            # Add epsilon to Tensorboard
            summary.add_scalar_value('Performance/epsilon', epsilon)
            if t_network.step % args.update_target_estimator_every == 0:
                t_network.load_state_dict(q_network.state_dict())
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(t, t_network.step, i_episode + 1, args.num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step in the environment
            action_probs = policy(state, epsilons[min(t_network.step, args.epsilon_decay_steps - 1)])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)  # initially take random action
            next_state, reward, done, _ = env.step(args.actions[action])
            next_state = helper.state_processor(next_state)
            next_state = np.append(state[1:, :, :], np.expand_dims(next_state, 0), axis=0)

            # Save transition to replay memory
            replay_memory.add(Transition(state, action, reward, next_state, done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t


            # Sample a minibatch from the replay memory
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_memory.sample(args.batch_size)

            # Calculate q values and targets
            q_network.forward(state_batch)
            q_network.compute_action_pred(action_batch)

            q_value_next = t_network.forward(next_state_batch)
            t_max_value, t_action = torch.max(q_value_next, dim=1)
            t_values = reward_batch + np.invert(done_batch).astype(np.float32) * args.discount_factor * t_max_value.numpy() # remove reward of final state


            # Perform gradient descent update
            optimizer.zero_grad()
            loss = q_network.compute_loss(t_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

            if done:
                # game ended
                break

            state = next_state
            t_network.step += 1

        # Add summaries to tensorboard
        summary.add_scalar_value("episode_reward", stats.episode_rewards[i_episode])
        summary.add_scalar_value("episode_length", stats.episode_lengths[i_episode])
        summary.to_zip(summary_path)

        yield t_network.step, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])

    env.monitor.close()
    return stats