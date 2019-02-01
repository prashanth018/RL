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
            action_Array = np.zeros(nA)
            #print(action_Array)
            for actions in range(nA):
                action_val = 0
                for prob, next_state, reward, done in env.env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                    print(action_val)
                #print(action_val)
                action_Array[actions] = action_val
                extracted_policy[states] = np.argmax(action_Array)
        return extracted_policy.reshape((8, 8))





if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v0')
    nS = env.observation_space.n
    nA = env.action_space.n

    policy = np.ones((nS, nA)) / 4
    policy_evaluation(env, policy)
    print(get_policy_view(env, policy=policy))
