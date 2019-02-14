import numpy as np
import sys
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")

from lib.envs.blackjack import BlackjackEnv
from lib import plotting
from pprint import pprint

env = BlackjackEnv()


def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player score: {} (Usable ace: {}), Dealer score: {}".format(score, dealer_score, usable_ace))


def strategy(observation):
    score, dealer_score, usable_ace = observation
    if score >= 20:
        return 0
    else:
        return 1


# Using first occurrence MC prediction
def monte_carlo_prediction(env, num_eps, discount_factor=1.0):
    returns_sum = defaultdict(float)
    returns_counts = defaultdict(float)

    V = defaultdict(float)
    for _ in range(num_eps):
        observation = env.reset()
        episode = []

        for t in range(100):
            action = strategy(observation)
            next_state, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                break
            observation = next_state

        states_in_episode = set([x[0] for x in episode])
        index_in_episode = []
        for state in states_in_episode:
            for i in range(len(episode)):
                if episode[i][0] == state:
                    index_in_episode.append(i)
                    break

        for ind, state in enumerate(states_in_episode):
            if state not in returns_sum:
                returns_sum[state] = 0
            if state not in returns_counts:
                returns_counts[state] = 0
            returns_sum[state] += sum(
                [episode[i][2] * (discount_factor ** i) for i in range(index_in_episode[ind], len(episode))])
            returns_counts[state] += 1

            V[state] = returns_sum[state] / returns_counts[state]
    return V


if __name__ == "__main__":
    V_over500k = monte_carlo_prediction(env, num_eps=500000)
    pprint(V_over500k)
    plotting.plot_value_function(V_over500k, title="5,00,000 Steps")
