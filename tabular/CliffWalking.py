import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")

from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
from collections import defaultdict
from pprint import pprint


def fn_policy(Q, env, epsilon=0.05):
    nS = env.nS
    nA = env.nA
    policy = np.ones((nS, nA)) * epsilon / nA
    for states in range(nS):
        policy[states][np.argmax(Q[states])] += (1 - epsilon)
    return policy


def q_learning_cliff_walking(Q, env, eps=200, epsilon=0.05, alpha=0.5, discount_factor=1.0, timesteps=800):
    nS = env.nS
    nA = env.nA
    episode_reward = []
    episode_length = []
    init_epsilon = epsilon
    for ep in range(eps):
        epsilon -= (init_epsilon * 0.0005)
        epsilon = max(epsilon, 0)
        policy = fn_policy(Q, env, epsilon=epsilon)
        current_state = env.reset()
        current_action = np.random.choice(np.arange(nA), p=policy[current_state])
        total_reward = 0
        for ts in range(timesteps):
            next_state, reward, done, prob = env.step(current_action)
            total_reward += reward
            next_action = np.random.choice(np.arange(nA), p=policy[next_state])
            next_greedy_action = np.argmax(policy[next_state])
            Q[current_state][current_action] = Q[current_state][current_action] + alpha * (
                    reward + (discount_factor * Q[next_state][next_greedy_action]) - Q[current_state][current_action])
            next_action_array = np.ones(nA) * epsilon / nA
            next_action_array[np.argmax(Q[current_state])] += (1 - epsilon)
            policy[current_state] = next_action_array
            if done:
                print("Episode {} ended after {} timesteps with total reward of {}".format(ep, ts, total_reward))
                episode_length.append(ts)
                episode_reward.append(total_reward)
                break
            current_state = next_state
            current_action = next_action
    stats = plotting.EpisodeStats(episode_lengths=episode_length, episode_rewards=episode_reward)
    plotting.plot_episode_stats(stats)


def sarsa_cliff_walking(Q, env, eps=200, epsilon=0.05, alpha=0.5, discount_factor=1.0, timesteps=800):
    nS = env.nS
    nA = env.nA
    episode_reward = []
    episode_length = []
    for ep in range(eps):
        policy = fn_policy(Q, env, epsilon=epsilon)
        current_state = env.reset()
        current_action = np.random.choice(np.arange(nA), p=policy[current_state])
        total_reward = 0
        for ts in range(timesteps):
            next_state, reward, done, prob = env.step(current_action)
            total_reward += reward
            next_action = np.random.choice(np.arange(nA), p=policy[next_state])
            Q[current_state][current_action] = Q[current_state][current_action] + alpha * (
                    reward + (discount_factor * Q[next_state][next_action]) - Q[current_state][current_action])
            next_action_array = np.ones(nA) * epsilon / nA
            next_action_array[np.argmax(Q[current_state])] += (1 - epsilon)
            policy[current_state] = next_action_array
            if done:
                print("Episode {} ended after {} timesteps with total reward of {}".format(ep, ts, total_reward))
                episode_length.append(ts)
                episode_reward.append(total_reward)
                break
            current_state = next_state
            current_action = next_action
    stats = plotting.EpisodeStats(episode_lengths=episode_length, episode_rewards=episode_reward)
    plotting.plot_episode_stats(stats)


if __name__ == "__main__":
    env = CliffWalkingEnv()
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # q_learning_cliff_walking(Q, env, eps=2000, epsilon=0.9)
    sarsa_cliff_walking(Q, env, eps=800, epsilon=0.001)
    pprint(Q)
