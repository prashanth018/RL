import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

if "../" not in sys.path:
    sys.path.append("../")


def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """

    V = np.zeros(100)
    policy = {_: (np.ones(min(_, 100 - _) + 1) / (min(_, 100 - _) + 1)) for _ in range(1, 100)}
    ct = 0
    while True:
        prev_policy = copy.deepcopy(policy)
        ct += 1
        print(ct)
        for states in range(1, 99):
            action_Array = np.zeros(min(states, 100 - states) + 1)
            for actions in range(min(states, 100 - states) + 1):
                reward_head = 0
                reward_tail = 0

                if states + actions == 100:
                    reward_head = 100
                if states - actions == 0:
                    reward_tail = 0

                if states + actions == 100:
                    action_val = p_h * (reward_head + discount_factor * 0) + (1 - p_h) * (
                            reward_tail + discount_factor * V[states - actions])
                elif states - actions == 0:
                    action_val = p_h * (reward_head + discount_factor * V[states + actions]) + (1 - p_h) * (
                            reward_tail + discount_factor * 0)
                else:
                    action_val = p_h * (reward_head + discount_factor * V[states + actions]) + (1 - p_h) * (
                            reward_tail + discount_factor * V[states - actions])

                action_Array[actions] = action_val
            V[states] = np.max(action_Array)
            policy[states] = np.eye(min(states, 100 - states) + 1)[np.argmax(action_Array)]

        if np.all(np.array([(policy[_] == prev_policy[_]).all() for _ in policy.keys()])):
            return policy, V


policy, V = value_iteration_for_gamblers(0.25)

print("Reshaped Grid Value Function:")
print(V.reshape((10, 10)))
print("")

# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# for _ in policy.keys():
#     print(str(_) + " : " + policy[_])
# print("")
