import gym
import numpy as np
import sys
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")

from lib.envs.windy_gridworld import WindyGridworldEnv


def fn_policy(env, Q, epsilon=0.05):
    nA = env.nA
    nS = env.nS
    policy = np.ones((nS, nA)) * epsilon / nA
    for state in range(nS):
        action_array = Q[state]
        policy[np.argmax(action_array)] += (1 - epsilon)
    return policy


def sarsa_WindyGridWorld(Q, env, discount_factor=1.0, alpha=0.05, epsilon=0.05):
    episodes = 200
    nS = env.nS
    nA = env.nA
    for ep in range(episodes):
        timesteps = 200
        current_state = env.reset()
        policy = fn_policy(env, Q)
        for ts in range(timesteps):
            action_arr = policy[current_state]
            action = np.random.choice(np.arange(action_arr), p=action_arr)
            next_state, reward, done, prob = env.step(action)
            next_action_arr = policy[next_state]
            next_action = np.random.choice(np.arange(next_action_arr), p=next_action_arr)
            Q[current_state][action] = Q[current_state][action] + alpha * (
                    reward + (discount_factor * Q[next_state][next_action]) - Q[current_state][action])
            act_arr = np.ones((nS, nA)) * epsilon / nA
            act_arr[np.argmax(Q[current_state])] += (1-epsilon)
            policy[current_state] = act_arr


if __name__ == "__main__":
    # for i in range(15):
    #     print(env.step(1))
    #     env.render()

    env = WindyGridworldEnv()
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    sarsa_WindyGridWorld(Q, env)
