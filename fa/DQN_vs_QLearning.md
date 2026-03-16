# Reflecting: DQN vs Q-Learning

## Why DQN Was Needed

Q-learning is tabular and keeps track of individual states and corresponding actions. This is not feasible for a continuous state space. DQN addresses this by using a function approximator (a neural network) to generalize across states.

---

## Problems DQN Solves

### Problem 1: Sample Correlation

In online RL, consecutive experiences are highly correlated — if you're walking down a corridor, the next 50 states are all "corridor." Training a neural network on correlated sequential data violates the IID assumption that SGD relies on, causing the network to overfit to recent experience and forget older knowledge.

**Solution: Experience Replay Buffer**

Store transitions `(s, a, r, s', done)` in a buffer. Sample random minibatches for training. This breaks temporal correlation — your batch might contain one transition from episode 5, another from episode 50, another from episode 100.

---

### Problem 2: Moving Target

In Q-learning, the target `r + γ · max Q(s', a'; θ)` depends on the same parameters `θ` being updated. Every gradient step changes the target being chased — like a dog chasing its own tail. This causes oscillations or divergence in the optimization landscape.

**Solution: Target Network**

Maintain a separate copy of the network with parameters `θ⁻`. Use this frozen copy to compute targets. Only update `θ⁻` periodically (either hard copy every N steps, or soft update via Polyak averaging `θ⁻ ← τθ + (1-τ)θ⁻`). The target is now stable between updates, giving the online network a fixed objective to chase.

---

## Problem DQN Introduces: Maximization Bias

The fix for Problem 2 contains a `max` term in the Bellman target:

```
r + γ · max Q(s', a'; θ⁻)
```

Two operations are happening here:
1. Evaluation of Q-values for next states by the target network
2. Picking the max of those values

### Why This Is a Problem

Say you're in state `s'` with 3 possible actions, and the true Q-values are all equal:

```
Q*(s', a₁) = 10.0
Q*(s', a₂) = 10.0
Q*(s', a₃) = 10.0
```

The true max is `10.0`. But the network's estimates are noisy. On a particular forward pass:

```
Q(s', a₁; θ⁻) = 11.2    (overestimates by 1.2)
Q(s', a₂; θ⁻) = 8.7     (underestimates by 1.3)
Q(s', a₃; θ⁻) = 10.5    (overestimates by 0.5)
```

The `max` operator picks `a₁` with value `11.2`. The true value is `10.0`, so the target is overestimated by `1.2`.

This is **systemic**: the `max` operator preferentially selects whichever action has the highest positive noise. Even if estimation errors are zero-mean (equally likely to be positive or negative), taking the max over those errors always skews positive. Run this experiment 1000 times with different random noise, and the expected value of `max_a Q(s', a; θ⁻)` will always be `≥ 10.0`, never below. The more actions you have, or the noisier your estimates, the worse the overestimation.

> **Note:** The overestimation/underestimation isn't the devil issue. The consistent selection of overestimated values due to the `max` operation is the real problem.

### Cascading Effect

These overestimated targets become training data. The online network trains toward `y = r + γ · 11.2` instead of the correct `r + γ · 10.0`. This makes Q-values too high, which feeds into future targets, pushing values even higher — a positive feedback loop where Q-values inflate over training, sometimes exploding to values that are physically impossible for the environment.

In practice, this can lead to Q-values drifting upwards more and more, increasing the instability of the learned policy.

---

## Double DQN (DDQN): the fix

Decouple the two operations — **action selection** and **action evaluation** by using two different networks for the two jobs:

- **Action selection**: use the online network `θ` to pick the best action
- **Action evaluation**: use the target network `θ⁻` to evaluate its Q-value


```
a* = argmax_a Q(s', a; θ)          ← Online network SELECTS
y  = r + γ · Q(s', a*; θ⁻)         ← Target network EVALUATES
```

The online network picks which action it thinks is best. The target network says how much that action is actually worth. Because the two networks have different parameter values (and therefore different noise patterns), the overestimation bias is broken.

```
y = r + γ · Q(s', argmax_a Q(s', a; θ); θ⁻)
```

With the same true values & same target network estimates. But now the online network has its own noise:

```
Online network θ:
Q(s', a₁; θ) = 9.3
Q(s', a₂; θ) = 10.8    ← Online picks a₂ (highest in its estimate)
Q(s', a₃; θ) = 10.1

Target network θ⁻:
Q(s', a₁; θ⁻) = 11.2
Q(s', a₂; θ⁻) = 8.7    ← Target evaluates a₂
Q(s', a₃; θ⁻) = 10.5
```

The online network selects a₂ (its highest). But the target network evaluates a₂ at 8.7. So the target is y = r + γ · 8.7, which underestimates by 1.3.

Run this many times across different noise patterns: sometimes you overestimate, sometimes you underestimate, but the systematic upward bias is gone because the network that picks the action isn't the same one that scores it. The noise is decorrelated.
