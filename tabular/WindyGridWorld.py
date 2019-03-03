import gym
import numpy as np
import sys
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")

from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
from pprint import pprint


def fn_policy(env, Q, epsilon=0.05):
    nA = env.nA
    nS = env.nS
    policy = np.ones((nS, nA)) * epsilon / nA
    for state in range(nS):
        action_array = Q[state]
        policy[state][np.argmax(action_array)] += (1 - epsilon)
    return policy


def sarsa_WindyGridWorld(Q, env, eps=200, discount_factor=1.0, alpha=0.5, epsilon=0.05):
    nS = env.nS
    nA = env.nA
    episode_lengths = []
    episode_rewards = []
    for ep in range(eps):
        timesteps = 700
        current_state = env.reset()
        policy = fn_policy(env, Q, epsilon=epsilon)
        action_arr = policy[current_state]
        action = np.random.choice(np.arange(nA), p=action_arr)
        total_reward = 0
        for ts in range(timesteps):
            next_state, reward, done, prob = env.step(action)
            total_reward += reward
            next_action_arr = policy[next_state]
            next_action = np.random.choice(np.arange(nA), p=next_action_arr)
            Q[current_state][action] = Q[current_state][action] + alpha * (
                    reward + (discount_factor * Q[next_state][next_action]) - Q[current_state][action])
            act_arr = np.ones(nA) * epsilon / nA
            act_arr[np.argmax(Q[current_state])] += (1 - epsilon)
            policy[current_state] = act_arr
            if done:
                print("Episode {} ended after {} timesteps".format(ep, ts))
                episode_lengths.append(ts)
                episode_rewards.append(total_reward)
                break
            current_state = next_state
            action = next_action

    stats = plotting.EpisodeStats(episode_lengths=np.array(episode_lengths), episode_rewards=np.array(episode_rewards))
    plotting.plot_episode_stats(stats)


if __name__ == "__main__":
    env = WindyGridworldEnv()
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    sarsa_WindyGridWorld(Q, env, eps=800)
    pprint(Q)
