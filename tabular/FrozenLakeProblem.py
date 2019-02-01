import gym
import numpy as np
import copy


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
                    print(action_val)
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


if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v0')
    nS = env.observation_space.n
    nA = env.action_space.n

    policy = np.ones((nS, nA)) / 4
    optimal_V, optimal_policy = policy_iteration(env)
    print("Grid Policy (0=up, 1=right, 2=down, 3=left)")
    print(get_policy_view(env, policy=optimal_policy))
    print(optimal_V.reshape((8, 8)))

    for episodes in range(5):
        observation = env.reset()
        print(observation)
        for t in range(300):
            # env.render()
            action = np.argmax(optimal_policy[observation])
            print("action at time {} = {}".format(t + 1, action))
            next_step, reward, done, info = env.step(action)
            print(next_step, reward, done, info)
            if done:
                if reward == 0:
                    print("LOSE!!")
                else:
                    print("WIN!!")
                print("Episode done after {} timesteps".format(t + 1))
                print()
                break
            observation = next_step

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
