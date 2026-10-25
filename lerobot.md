# Robot Learning FAQ

## The Shift to Robot Learning

### What does it mean that "learning can prove just as effective as explicit modeling"?

**Explicit Modeling (Traditional):** Engineers had to manually code complex physics equations for rigid-body dynamics, balance, and contact modeling.

**Learning (Modern AI):** Neural networks learn how to perform physical tasks (like walking or grabbing) implicitly through data, trial, and error.

**Takeaway:** Using data to teach a robot *how* to move is proving highly effective and much less labor-intensive than manually coding real-world physics from scratch.

### How do learning-based approaches for "whole-body control" differ from "normal" (classical) locomotion?

**Implicit vs. Explicit:** Classical robotics relies on strict mathematical calculations (like center of mass). Learning lets the AI figure out balance intuitively.

**Whole-Body vs. Modular:** Classical systems often separate leg movement from arm movement. Learning controls every joint simultaneously, allowing the robot to use its arms to counterbalance a stumble.

**Fluid vs. Stiff:** Classical models use strict safety margins, leading to stiff walking. Learning generates "rich," fluid, and highly dynamic motion that handles unpredictable terrain better.

### Why are both task-specific models and generalist models grouped under "Behavioral Cloning (BC)"?

Even though generalist models (like the ChatGPTs of robotics) are language-conditioned and task-specific models are not, they are categorized together because they share the same fundamental training method: they both learn by mimicking massive datasets of human demonstrations (expert trajectories).

---

## LeRobot Architecture & Data

### Is the lerobot inference stack based out of Rust?

No. It is a pure Python/PyTorch stack. It keeps inference (model prediction) and control (hardware actuation) separated into a Policy Server and Robot Client that communicate asynchronously via gRPC. This keeps the barrier to entry low for Python-based AI researchers.

### Why merge millions of trajectories into the "same high-level structure" for scalability?

Robotics datasets can contain billions of camera frames. Saving each attempt as a separate file would overwhelm file systems and cloud storage. To solve this, `lerobot` concatenates the footage into large chunks (like large `.mp4` or `.parquet` files) and uses JSON metadata files to index the start and end points of specific episodes.

### What is `meta/stats.json` calculating?

It stores global statistics (mean, standard deviation, min, max) for each individual feature (like joint states or actions) calculated across the **entire dataset**, not just a single episode. This is primarily used to normalize data during model training.

---

## Dimensionality & Data Collection

### How does the dimensionality work when fetching `delta_timestamps` like `[-0.2, -0.1, 0.0]` with a batch size of 16?

It results in a 5D tensor: `(16, 3, C, H, W)`.

* `C, H, W`: The 3D tensor of a single image (Channels, Height, Width).
* `3`: The time window, representing the 3 *discrete* frames grabbed at those specific offsets, making a 4D sample: `(3, C, H, W)`.
* `16`: The DataLoader batch size prepended to the front.

*(Note: To get every continuous frame in a window, you must explicitly list every time offset based on the dataset's FPS).*

### In the data collection example, how does "teleop" work, and where do the states/actions come from?

**Camera:** A standard webcam (via OpenCV) captures the visual stream.

**Actions:** The human physically moves a **Leader** arm. The angles from this arm are recorded as the "expert commands" (actions).

**States:** The **Follower** arm mimics the Leader. The physical sensors on the Follower arm record the actual joint angles achieved (states).

Everything is synchronized and recorded in discrete ticks (e.g., 30 FPS).

### What hardware is required, and what happens if you move too fast during data collection?

**Hardware:** You can build an open-source SO-100 dual-arm setup (3D printed parts + smart servos) for ~$450-$500. Alternatively, you can do this entirely in simulation without physical hardware.

**Resolution Limits:** If you move too fast, you hit the 30 FPS limit (losing data between ticks), cause visual motion blur, exceed the motor's physical torque/speed limits, and bottleneck the serial bus. Smooth, deliberate movement is required.

---

## Kinematics (FK, IK, and Diff-IK)

### What is the difference between Forward Kinematics (FK) and Inverse Kinematics (IK)?

**FK:** You know the **joint angles**, and you calculate where the **end effector (hand)** will end up.

**IK:** You know where you want the **end effector** to go, and you calculate what **joint angles** are required to reach that target.

### Why is the end effector computation $l \cos(\theta_1 + \theta_2)$ instead of just $l \cos(\theta_2)$?

In standard robot designs, the second joint's angle ($\theta_2$) is measured relative to the *first link*, not the global base. To find its absolute angle relative to the ground, you must add the rotation of the first link ($\theta_1 + \theta_2$).

### What does "expert-defined trajectories obviate this problem providing a length-K succession of goal poses" mean?

Giving standard IK a single final target destination can result in the robot taking an unpredictable or dangerous path to get there. Experts solve this by designing a smooth trajectory broken down into $K$ intermediate checkpoints. The robot connects the dots, ensuring a safe, predictable movement.

### What is Differential IK (Diff-IK), and why does it replace standard IK for following trajectories?

Calculating standard absolute IK positions for thousands of checkpoints is too slow for real-time movement. Diff-IK calculates **velocities** instead of positions. Using a matrix called the Jacobian ($J$), it asks: *"If the hand needs to move this fast in this direction, how fast should the joints spin right now?"* This linear math shortcut allows the robot to fluidly trace complex paths in real-time.

---

## Reinforcement Learning Algorithms & Noise

### Why is SAC (Soft Actor-Critic) considered a "strong" algorithm?

1. **Maximum Entropy:** It optimizes for both reward and randomness. By keeping options open, it learns multiple ways to succeed, resulting in highly robust policies that don't easily break.
2. **Sample Efficiency:** It is an "off-policy" algorithm, meaning it stores and learns from every past mistake and success in a replay buffer, requiring far fewer physical attempts to learn.
3. **Continuous Action Spaces:** It was built from the ground up to output fluid, continuous numbers (like motor voltages) rather than discrete game-like button presses.

### Is Domain Randomization (e.g., randomizing friction) using Dirichlet noise?

No. Physical properties (friction, mass) are independent quantities, so engineers use **Uniform** or **Gaussian** distributions to randomize them. Dirichlet noise is strictly used for probabilities that must sum to exactly 1 (100%).

### What about injecting noise to make policies robust to real-world actuator errors? Is that Dirichlet?

That concept is called **Actuator Noise Injection** (or Target Policy Smoothing), but it still does *not* use Dirichlet noise because motor commands (like degrees or volts) are continuous scalars that don't sum to 1. Engineers use Gaussian or Ornstein-Uhlenbeck (momentum-based) noise for this.

*When to use Dirichlet:* Dirichlet noise is used when an AI outputs a **discrete probability vector** (e.g., a 70% chance to move forward, 30% chance to turn). Board game AIs like AlphaZero use Dirichlet noise to scramble these vectors and force exploration.
## Training & Eval

### Dataset
- `LeRobotDataset` has two layers. `ds.hf_dataset` is a plain Arrow table with the numeric columns only (`action`, `observation.state`, `episode_index`, `frame_index`, `timestamp`, `next.done`, `index`, `task_index`). Video lives separately as MP4s.
- `ds[i]` = row `i` of `hf_dataset` + the decoded video frame. That's why reading `hf_dataset` works without ffmpeg but `ds[i]` doesn't.
- `action` = where the human told the arm to go. `observation.state` = where the arm actually is. So `action[t] ≈ state[t+1]`.
- `frame_index` is the position within an episode, `index` is the global row. Episodes are contiguous, so `hf[0:400]` is episode 0 for transfer cube.
- `--dataset.episodes='[0,1,2]'` filters the dataset before sampling. On transfer cube that's 1,200 of 20,000 frames. This is generally used to smoke test. Expect the model to overfit to these 3 episodes.

### Config & flags
- Folder structure: `configs/train.py` (top level: `--steps`, `--batch_size`, `--num_workers`), `configs/default.py` (`--dataset.*`), `configs/policies.py` (`--policy.*` shared), `policies/<name>/configuration_<name>.py` (`--policy.*` per policy).
- `--output_dir` throws an exception unless `--resume=true`, else `FileExistsError`.

### Training logs
- `step` = optimizer updates. `smpl` = `step × batch_size`. `ep` = `smpl / frames_per_episode`. `epch` = `smpl / total_frames`.
- `grdn` = gradient norm, `√(sum of all grads²)`, i.e. the length of the "change everything" arrow. Measured *before* clipping. ACT policy applies a default clipping at `10.0`. That is, the direction is kept, but the update amplitude is shortened. Large early, want it to fall toward and below 10. 
    - A spike or sudden drop means gradient explosion/implosion.
- `updt_s` = GPU time spent to do forward + backward + `optimizer.step` (i.e., being useful). `data_s` = how long the training loop sat waiting on `next(dataloader)`(i.e. GPU being useless).
    - `data_s ≈ 0` is the goal. On a slow GPU, decoding & dataloader hides behind compute (bad problem to have). On a fast GPU, the CPU workers can fall behind (good problem to have).
    - If CPU workers are falling behind (data-bound): `workers_needed ≈ current_workers × (updt_s + data_s) / updt_s`, capped by CPU cores. `--num_workers` is top level, default 4.
- `--log_freq` lets you set the metric logging frequency. Default 200.

### Checkpoints & uploads
- Checkpoint = `pretrained_model/` (~197 MB: weights, `config.json`, pre/post processors) + `training_state/` (~394 MB: Adam state, 2× the weights). `training_state` is only needed to resume.
- `--policy.push_to_hub` defaults to `True` and needs `--policy.repo_id`, else it refuses to start. Fires once at the very end. Public by default unless `--policy.private=true`. Needs HF auth.
- W&B uploads a model artifact at every checkpoint unless `--wandb.disable_artifact=true`.

### Eval
- Mid-training eval needs `--env.type` + `--env.task`, and runs on the live model in memory. *No checkpoint needed to be saved for mid-training eval runs.* Logs `pc_success`, `avg_max_reward`, and one rollout video per eval to W&B. Default `eval_freq` is 20,000.
- `pc_success` = `mean(success) × 100`. `eval_s` = total wall clock eval time. `eval_ep_s` = `eval_s / n_episodes`.
- `avg_max_reward` = furthest stage reached, averaged. Transfer cube stages: 1 touch, 2 lift, 3 both grippers on it, 4 transferred. Way more useful than `pc_success` early on, when success sits at 0 for a long time.
- Evals run on CPU, so evals don't get faster with a better GPU.
- Headless cloud box: `export MUJOCO_GL=egl` to run evals without rendering.
