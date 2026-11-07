# Flow Matching From Scratch

A minimal flow matching implementation in PyTorch, trained on a toy problem: turn Gaussian noise into samples from a target distribution.

**TL;DR:** A small MLP learns a velocity field that carries `N(0, I)` noise to a 4-dim Gaussian. After 2750 steps the sampled means land within ~3% of the targets. Standard deviations match on dims 0 and 3 but come out too narrow on dims 1 and 2.

## What's here

- `flow_matching.py`: model, training loop, Euler sampler, plotting, checkpoint save/load.
- `plots/`: histograms of 256 samples from the model at the steps shown below, plus the loss curve. The script saves one every 250 steps.
- `checkpoints/`: trained weights.

## The toy problem

The target is a 4-dim Gaussian with independent dimensions:

| Dim | Mean | Std |
|---|---|---|
| 0 | 8.1 | 2.1 |
| 1 | 5.0 | 1.3 |
| 2 | 61.3 | 7.7 |
| 3 | 22.8 | 2.8 |

The model never sees these numbers. It only sees samples drawn from this distribution, the same way a robot policy only sees demos.

## How it works

**Training.** For every row in the batch, independently:

1. `x1`: a sample from the target (the "demo").
2. `x0 ~ N(0, I)`: noise.
3. `t ~ U(0, 1)`: a random time.
4. Point on the straight line from noise to sample: `xt = (1 - t) * x0 + t * x1`.
5. Regress the model's velocity onto that line's velocity: `loss = MSE(model(xt, t), x1 - x0)`.

There is no sampling loop during training, just one forward pass per batch.

**Sampling.** Start from noise and follow the learned velocity with Euler steps:

```
xt = noise
for i in range(N):
    xt = xt + model(xt, i / N) * (1 / N)
```

## Setup

| Setting | Value |
|---|---|
| Model | MLP, input `[xt, t]` (5) → 64 → 64 → 64 → 4, GELU |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 256 |
| Steps | 3000 (each "epoch" in the code is one gradient step on a fresh batch) |
| Sampling | 10 Euler steps, 256 samples |

## Results

**Step 0** (untrained): the samples are still just the noise.

![epoch 0](plots/epoch_00000.png)

**Step 250:** moving toward the targets, but means are only about half way (dim 2 at 33.1 of 61.3).

![epoch 250](plots/epoch_00250.png)

**Step 500:** means mostly there.

![epoch 500](plots/epoch_00500.png)

**Step 750:** spreads start to widen.

![epoch 750](plots/epoch_00750.png)

**Step 2750** (last saved):

![epoch 2750](plots/epoch_02750.png)

**Target vs learned at step 2750**

| Dim | Target mean | Learned mean | Target std | Learned std |
|---|---|---|---|---|
| 0 | 8.1 | 8.3 | 2.1 | 2.1 |
| 1 | 5.0 | 5.0 | 1.3 | 1.0 |
| 2 | 61.3 | 60.2 | 7.7 | 4.6 |
| 3 | 22.8 | 22.7 | 2.8 | 2.6 |

**Loss**

![loss curve](plots/loss_curve.png)

## Observations

- Means are learned first, spreads later. By step 500 every mean is within 5% of its target, but every std is still too narrow.
- The flow matching loss never goes to 0. Many (noise, sample) pairs pass through the same `(xt, t)` with different `x1 - x0`, so the best the model can do is predict their average. The loss flattens at a floor above zero.
- Dim 2 has the largest scale (mean 61.3, std 7.7) and is the furthest off on std (4.6). The loss has been flat around 7-8 since step 1750, and dim 2's std has stayed at 4.3-4.8 since step 1000, so training longer probably won't fix it. Unnormalized targets and only 10 Euler steps are the suspects. Not verified yet.

## Next

- *This target has a single mode per dimension*, so it tests the pipeline, not multimodality. Next run: the 70/40 gripper example from the [survey](../robot-learning-survey.md#various-outputs-and-losses-for-robot-policy) (`+1` with probability 70/110, else `-1`). MSE would output +0.27. Flow matching should produce two peaks.
- Normalize targets and try more Euler steps to close the std gap on dim 2.
