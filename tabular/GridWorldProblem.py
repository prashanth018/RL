import numpy as np
import sys
import copy

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()


# Tabular Policy Evaluation
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)

    while True:
        delta = 0
        for states in range(env.nS):
            val = 0
            for actions, action_prob in enumerate(policy[states]):
                # Here the assumption is that we get the reward after we reach the next
                # state(contrary to what we assume that we get reward after we take an action).
                for prob, next_state, reward, done in env.P[states][actions]:
                    val += action_prob * prob * (reward + discount_factor * (V[next_state]))

            delta = max(delta, np.abs(V[states] - val))
            V[states] = val
        if delta <= theta:
            break
    return np.array(V)


# Tabular Policy Iteration
def policy_iteration(env, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # Giving the current policy and getting it evaluated.
        V = policy_eval(policy, env, discount_factor)
        optimal = True

        # Uncomment the below code to see how the value and policy changes over iterations.
        # print("Reshaped Grid Value Function:")
        # print(V.reshape(env.shape))
        # print("")
        # print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
        # print(np.reshape(np.argmax(policy, axis=1), env.shape))
        # print("***********")


        for states in range(env.nS):
            # Since this is a model based environment, greedy approach
            # will work whereas this will not work on a model-free environment.
            # action_Array stores the action values for all the possible actions for a given state.
            action_Array = np.zeros(env.nA)
            for actions in range(env.nA):
                action_val = 0
                for prob, next_state, reward, done in env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                action_Array[actions] = action_val

            max_action_index = np.argmax(action_Array)
            current_policy_action_index = np.argmax(policy[states])

            if max_action_index != current_policy_action_index:
                optimal = False

            policy[states] = np.eye(env.nA)[max_action_index]

        if optimal:
            return policy, V

# Uncomment below code to see the output of policy iteration.
# policy, v = policy_iteration(env)
# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")
#
# print("Reshaped Grid Value Function:")
# print(v.reshape(env.shape))
# print("")


# Tabular Value Iteration
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        prev_policy = copy.deepcopy(policy)

        # Uncomment the below code to see how the value and policy changes over iterations.
        # print("Reshaped Grid Value Function:")
        # print(V.reshape(env.shape))
        # print("")
        # print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
        # print(np.reshape(np.argmax(policy, axis=1), env.shape))
        # print("***********")

        for states in range(env.nS):
            action_Array = np.zeros(env.nA)
            for actions in range(env.nA):
                action_val = 0
                for prob, next_state, reward, done in env.P[states][actions]:
                    action_val += prob * (reward + discount_factor * V[next_state])
                action_Array[actions] = action_val
            V[states] = np.max(action_Array)
            policy[states] = np.eye(env.nA)[np.argmax(action_Array)]
            # print(action_Array)
        if np.all(prev_policy == policy):
            return policy, V



# Uncomment below code to see the output of value iteration.
# policy,V = value_iteration(env)
# print("Reshaped Grid Value Function:")
# print(V.reshape(env.shape))
# print("")
#
# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")
