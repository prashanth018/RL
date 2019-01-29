import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

#Tabular Policy Evaluation
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

    # for states in range(env.nS):
    #     for actions in range(env.nA):
    #         print(states,actions,env.P[states][actions])

    while True:
        delta = 0
        for states in range(env.nS):
            val = 0
            for actions,action_prob in enumerate(policy[states]):
                # Here the assumption is that we get the reward after we reach the next
                # state(contrary to what we assume that we get reward after we take an action).
                for prob, next_state, reward, done in env.P[states][actions]:
                    val += action_prob*prob*(reward + discount_factor * (V[next_state]))

            delta = max(delta,np.abs(V[states] - val))
            V[states] = val
        if delta <= theta:
            break
    return np.array(V)


random_policy = np.ones((env.nS,env.nA)) / env.nA
v = policy_eval(random_policy,env)

print(v)
