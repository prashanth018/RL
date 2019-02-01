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
    while True:
        prev_policy = copy.deepcopy(policy)
        for states in range(1, 100):
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

def PlotCapitalvsValue(V):
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')

    plt.title('Final Policy (action stake) vs State (Capital)')

    x = range(100)
    y = V[:100]
    plt.plot(x, y)

    plt.show()



plt.xlabel('Capital')
plt.ylabel('Value Estimates')

plt.title('Final Policy (action stake) vs State (Capital) over varying Head Probabilities')

for prob_h in [0.2,0.4,0.6,0.8]:
    policy,V = value_iteration_for_gamblers(prob_h)
    x = range(100)
    y = V[:100]
    plt.plot(x, y,label=str(prob_h))

plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('CapitalvsValueEstimate.png', dpi=100)

