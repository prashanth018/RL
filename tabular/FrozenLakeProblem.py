import gym
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import sys


def policy_evaluation(env, policy, theta=0.0001, discount_factor=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n
    new_V = np.zeros(nS)
    while True:
        V = copy.deepcopy(new_V)
        for states in range(nS):
            action_Array = np.zeros(nA)
            for actions in range(nA):
                action_val = 0
                for prob, next_state, reward, done in env.env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                action_Array[actions] = action_val
            new_V[states] = np.dot(policy[states], action_Array)
        if np.max(np.abs(new_V - V)) < theta:
            return new_V


def get_policy_view(env, V=None, policy=None, discount_factor=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n
    if V is None and policy is None:
        print("No arguments passed")
        return
    elif policy is not None:
        return np.argmax(policy, axis=1).reshape(8, 8)
    elif V is not None:
        extracted_policy = np.zeros(nS)
        for states in range(nS):
            action_array = np.zeros(nA)
            # print(action_Array)
            for actions in range(nA):
                action_val = 0
                for prob, next_state, reward, done in env.env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                    # print(action_val)
                # print(action_val)
                action_array[actions] = action_val
                extracted_policy[states] = np.argmax(action_array)
        return extracted_policy.reshape((8, 8))


def policy_iteration(env, discount_factor=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n

    policy = np.ones((nS, nA)) / nA

    while True:
        V = policy_evaluation(env, policy)
        prev_policy = copy.deepcopy(policy)
        for states in range(nS):
            action_array = np.zeros(nA)
            for actions in range(nA):
                action_val = 0
                for prob, next_state, reward, done in env.env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                action_array[actions] = action_val
            policy[states] = np.eye(nA)[np.argmax(action_array)]

        if (prev_policy == policy).all():
            return V, policy


# Instead of using this method you can directly call value_iteration_epsilon_greedy with eps=0
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)

    while True:
        prev_V = copy.deepcopy(V)

        for states in range(nS):
            action_array = np.zeros(nA)
            for actions in range(nA):
                action_val = 0
                for prob, next_state, reward, done in env.env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                action_array[actions] = action_val

            V[states] = np.max(action_array)

        if np.max(np.abs(prev_V - V)) <= theta:
            return V, get_policy_view(env, V=V)


def value_iteration_epsilon_greedy(env, eps=0.05, theta=0.0001, discount_factor=1.0):
    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)

    while True:
        prev_V = copy.deepcopy(V)

        for states in range(nS):
            action_array = np.zeros(nA)
            for actions in range(nA):
                action_val = 0
                for prob, next_state, reward, done in env.env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                action_array[actions] = action_val

            if eps == 0:
                V[states] = np.max(action_array)
            else:
                max_action = np.argmax(action_array)
                action_val = np.sum(action_array)
                V[states] = ((eps / nA) * action_val) + ((1 - eps) * action_array[max_action])

        if np.max(np.abs(prev_V - V)) <= theta:
            return V, get_policy_view(env, V=V)


def which_action(act):
    if act == 0:
        return "up"
    elif act == 1:
        return "right"
    elif act == 2:
        return "down"
    elif act == 3:
        return "left"


def epsilon_iteration():
    val = 100
    epsilons = list(np.array(range(val)) / 100)
    epsilon_data = {}
    for epsilon in epsilons:
        print("epsilon: {}".format(epsilon))
        epsilon_data[epsilon] = []
        optimal_V, policy_view = value_iteration_epsilon_greedy(env, eps=epsilon)
        episode_data = []
        policy_view = policy_view.reshape(64)

        for episodes in range(1000):
            observation = env.reset()
            for t in range(500):
                action = policy_view[observation]
                next_state, reward, done, info = env.step(action)
                if done:
                    if reward == 0:
                        # print("LOSE!!")
                        episode_data.append((t + 1, "lost"))
                    else:
                        # print("WIN!!")
                        episode_data.append((t + 1, "won"))
                    # print("Episode done after {} timesteps".format(t + 1))
                    # print()
                    break
                observation = next_state

        for episode in episode_data:
            if episode[1] == "won":
                epsilon_data[epsilon].append(episode[0])

    # Plot: 'epsilon vs average timesteps for win'
    x = epsilon_data.keys()
    y = [sum(epsilon_data[i]) / len(epsilon_data[i]) for i in x]

    fig = plt.figure()
    plt.xlabel('Epsilon')
    plt.ylabel('Avg. Timesteps')
    plt.title('Epsilon vs Avg. Timesteps taken for a win')

    plt.plot(x, y)
    plt.show()
    plt.draw()
    fig.savefig('EpsilonVsAvgTimesteps.png', dpi=100)

    # Plot: 'epsilon vs lost games in 1000 episodes'
    y = [1000 - len(epsilon_data[i]) for i in x]

    fig = plt.figure()
    plt.xlabel('Epsilon')
    plt.ylabel('Number of Loses over 1000 games')
    plt.title('Epsilon vs Number of Loses')

    plt.plot(x, y)
    plt.show()
    plt.draw()
    fig.savefig('EpsilonVsNumberOfLoses.png', dpi=100)


if __name__ == "__main__":

    env = gym.make('FrozenLake8x8-v0')
    nS = env.observation_space.n
    nA = env.action_space.n
    epsilon = 0.05
    policy = np.ones((nS, nA)) / 4
    config = np.array(range(64)).reshape((8, 8))

    optimal_V, optimal_policy = None, None
    policy_view = None
    algo = 'EpsilonIteration'

    if algo == 'PolicyIteration':
        optimal_V, optimal_policy = policy_iteration(env)
    elif algo == 'ValueIteration':
        optimal_V, policy_view = value_iteration(env)
    elif algo == 'ValueIterationEpsilonGreedy':
        optimal_V, policy_view = value_iteration_epsilon_greedy(env, eps=epsilon)
    elif algo == 'EpsilonIteration':
        epsilon_iteration()
        sys.exit()

    print("Grid Policy (0=up, 1=right, 2=down, 3=left)")
    if policy_view is None:
        policy_view = get_policy_view(env, policy=optimal_policy)
        print(policy_view)
    elif optimal_policy is None:
        print(policy_view)

    print(optimal_V.reshape((8, 8)))
    print(config)
    episode_data = []
    policy_view = policy_view.reshape(64)

    for episodes in range(1000):
        print("Episode No. {}".format(episodes))
        observation = env.reset()
        for t in range(500):
            # env.render()
            # action = np.argmax(optimal_policy[observation])
            action = policy_view[observation]
            # print(action)
            # print("action at time {} for observation {} is {}".format(t + 1, observation, which_action(action)))
            next_state, reward, done, info = env.step(action)
            # print(next_state, reward, done, info)
            # print(config)
            # print()
            if done:
                if reward == 0:
                    # print("LOSE!!")
                    episode_data.append((t + 1, "lost"))
                else:
                    # print("WIN!!")
                    episode_data.append((t + 1, "won"))
                # print("Episode done after {} timesteps".format(t + 1))
                # print()
                break
            observation = next_state

    x = range(1000)
    y = [tup[0] for tup in episode_data]
    label = []
    for tup in episode_data:
        if tup[1] == 'won':
            label.append(0)
        else:
            label.append(1)
    colors = ["blue", "red"]

    fig = plt.figure()

    plt.xlabel('Timesteps')
    plt.ylabel('Episodes')
    plt.title('Number of Timesteps per Episode')

    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.legend()
    plt.show()
    plt.draw()
    fig.savefig('EpisodesVsNumberOfTimesteps_VI.png', dpi=100)

# Policy Iteration
# Output of the value function
# [[0.99840141 0.99847128 0.99857219 0.99868219 0.99879309 0.99889915 0.99899398 0.99906337]
#  [0.99838398 0.99843718 0.99852459 0.99862888 0.99873978 0.99885244 0.99896855 0.99910431]
#  [0.9970193  0.97543691 0.92405708 0.         0.8554324  0.94510196 0.98107031 0.99918439]
#  [0.99578228 0.93097281 0.79831174 0.47358747 0.62250702 0.         0.94386811 0.99930012]
#  [0.99472455 0.82185388 0.54002386 0.         0.53854924 0.61052307 0.85126925 0.99944643]
#  [0.99389016 0.         0.         0.1676511  0.38265186 0.44177958 0.         0.99961694]
#  [0.99331384 0.         0.19332779 0.12031804 0.         0.33218493 0.         0.99980419]
#  [0.99301956 0.72625917 0.45972201 0.         0.27739063 0.55478456 0.77739063 0.        ]]

# Output of the Policy
# [[3 2 2 2 2 2 2 2]
#  [3 3 3 3 3 3 3 2]
#  [0 0 0 0 2 3 3 2]
#  [0 0 0 1 0 0 2 2]
#  [0 3 0 0 2 1 3 2]
#  [0 0 0 1 3 0 0 2]
#  [0 0 1 0 0 0 0 2]
#  [0 1 0 0 1 2 1 0]]


# Value Iteration
# Output of the value function
# [[0.99870022 0.99877314 0.99885578 0.99894108 0.99902471 0.99910354 0.99917354 0.99922463]
#  [0.99870654 0.99876529 0.99884024 0.99892201 0.99900533 0.99908818 0.99917284 0.99927189]
#  [0.99771314 0.97611533 0.92466011 0.         0.8557119  0.94535729 0.98128813 0.99934779]
#  [0.99682864 0.931921   0.79906523 0.47395214 0.62280773 0.         0.94405128 0.99944865]
#  [0.99608428 0.82287329 0.54064617 0.         0.5387708  0.61070526 0.85143916 0.99956987]
#  [0.99550541 0.         0.         0.16776494 0.38281741 0.44192354 0.         0.9997061 ]
#  [0.99511065 0.         0.19371585 0.12049359 0.         0.332251   0.         0.99985151]
#  [0.99491137 0.72774944 0.4607091  0.         0.27741419 0.55483155 0.77741525 0.        ]]

# Output of the Policy
# [[1 2 2 2 2 2 2 2]
#  [3 3 3 3 3 3 3 2]
#  [0 0 0 0 2 3 3 2]
#  [0 0 0 1 0 0 2 2]
#  [0 3 0 0 2 1 3 2]
#  [0 0 0 1 3 0 0 2]
#  [0 0 1 0 0 0 0 2]
#  [0 1 0 0 1 2 1 0]]
