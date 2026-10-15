# Robot Learning Survey

Working notes in Q&A form. `TO DO` / `need to confirm` markers are things I flagged as not yet understood.

**Contents**
1. [Behavior cloning basics](#behavior-cloning-basics)
2. [BC in LLM terms](#bc-in-llm-terms)
3. [Covariate shift](#covariate-shift)
4. [Various outputs and losses for robot policy](#various-outputs-and-losses-for-robot-policy)
5. [RL on real robots, and sim to real](#rl-on-real-robots-and-sim-to-real)
6. [Robot policy learning approaches](#robot-policy-learning-approaches)
7. [Where does the data come from?](#where-does-the-data-come-from)
8. [ACT](#act)
9. [VAE](#vae)
10. [Observations](#observations)
    - [LLM mid-training is imitation learning](#llm-mid-training-is-imitation-learning)
11. [Glossary](#glossary)
12. [Questions](#questions)

---

## Behavior cloning basics

### Basic Behavior Cloning

**Basic Behavior Cloning** involves collecting the expert `observations, action`. Then training the robot policy to predict action given observations. Loss involves minimizing the negative log likelihood of the policy where steps are sampled from the expert observations. The modern observation is roughly `o = (camera images, proprioception, language instruction)`.

### What is Proprioception?

Human brain has a perception of the body, limbs & surroundings. This makes it possible for humans to touch their nose with their finger tip with eyes closed. A robot needs to know its own current state to be able to predict the next action. Therefore, the **"proprioception"** in the observation.

### Where do actions come from?

Actions and the Proprioception come from **teleoperation**. A robot mimics the expert behavior, it is easy to get the camera images and language instruction but not the perception of the robot's own position and the action space. Teleoperation helps with getting the actions and perception in robot's own action space. This way, it collects the right actions corresponding to the given observation.

### Why are observations stacked?

To construct a **Markovian state** from the observations. This is because, plainly from the "velocity of actuator 4", it is hard to construct the gradient of the velocity / trajectory of actuator 4. Stacking observations helps construct a history.

### Learning behavior based on length of history

In BC, more history can make things worse. The model can learn to copy its previous action, because in expert data `a_t ≈ a_{t-1}` is almost always true. This is called the **copycat problem** or **causal confusion**. It's one reason many modern policies use short history.

---

## BC in LLM terms

### What is BC formally equivalent to, both in classic ML terms and in the LLM world?

In classic ML terms it's supervised learning (labelled data). In LLM terms it's `SFT with teacher forcing`: the expert's choice at each step is the label.

### What is SFT with teacher forcing?

SFT with teacher forcing is mathematically equivalent to minimizing the **forward KL**. It is **mass covering**. This means the student must assign a mass where teacher's P(x) > 0. SFT is a coarse approximation because you only sample a single hard token path from the teacher, i.e., you calculate the loss for one-hot output token.

### Is SFT with teacher forcing same as Knowledge Distillation?

In contrast, true **Knowledge Distillation** minimizes the forward KL divergence over the entire soft logit distribution at every single token step, forcing the student to match the teacher's exact shape, rather than just covering the specific text paths the teacher happened to sample. This helps transfer the broader knowledge, reasoning, or "dark matter" (as Hinton called it) of a larger model to a smaller one.

> SFT with Teacher Forcing = Learning What to Say.
> Knowledge Distillation = Learning How to Think.

---

## Covariate shift

### What is Covariate Shift? How is it different from Concept Drift & Prior Probability Shift? How are each of these handled in production?

**TO DO:** Deep dive on this, I couldn't understand the probabilistic reasoning. I could understand that Covariate Shift = model seeing data that it hasn't seen in its training data .. but I can't understand the mathematical reasoning behind it.

### Problem with Vanilla BC

Coming back to BC, you train the policy on expert observations. On held-out demo data the policy picks the expert's action correctly 99% of the time per step. You deploy it on a 500-step task, and it fails on most rollouts, not 1% of them. Why? What's different about the states it sees at test time compared with training?

**Covariate Shift**. At test time, it makes one small mistake and ends up slightly off the expert's path. It has never seen this state, so its next mistake is larger, which pushes it further off-distribution, and so on. Even if model was 99% right with its prediction each step, `0.99^500 ≈ 0.7%`, so 99% per-step accuracy was never going to mean 99% task success. Errors compound. Ross & Bagnell formalized this: with per-step error ε over horizon T, ordinary supervised learning gives total cost `O(εT)`, but BC gives `O(εT²)`. The extra factor of T comes from the fact that once you've drifted, you stay drifted for the rest of the episode. (I don't fully understand why, needs exploring later).

### How do we address the Covariate Shift problem?

**DAgger (Dataset Aggregation)**: Train the network on the human's driving. Let the network drive. It drifts left. The human watches and, at each moment, says what they would do: "here, steer right." Add those `(image, correct steering)` pairs to the dataset and retrain.

### Extending the Covariate Shift problem to LLMs, how do GPT models end up correcting themselves if I, say, type sentences with bunch of spelling mistakes?

Like say, 25 word sentence with spelling mistakes in all the 25 words and GPT still ends up understanding the meaning and context. Obviously, it's impossible to observe this exact combination in train data, but still GPTs seem to overcome this covariate shift during test time. How?

GPT was trained on trillions of words from millions of people: Reddit posts, texts, forum comments, all full of misspellings. It has seen "teh," "recieve," and "definately" thousands of times, usually next to text that makes clear what was meant. It never saw our exact 25-typo sentence, but it saw every kind of typo we made, so it learned that "teh" means "the". The robot's data comes from one or a few skilled people who almost never make mistakes. It has zero examples of the car 40 cm off-center. So the difference isn't that GPT is smarter. GPT's training data contains plenty of messy situations, and the robot's doesn't. If the robot's demos included lots of drifting followed by recovery, it would be fine too. (That's what DAgger provides.)

---

## Various outputs and losses for robot policy

- **MSE head:** Assuming we have n actuators. In this approach each actuator produces a continuous output like, `action = [joint1: +0.12, joint2: -0.30, joint3: +0.05, ..., gripper: 0.8]`.
- **Gaussian head (mean, variance per actuator):** In this approach each actuator produces a mean and variance. We then calculate the probability `z` vector for each actuator from the mean and variance. We compute this based on the expert action value (need to confirm this). We then take negative log likelihood on the prob vector for each actuator to compute the loss (need to confirm). The advantage over the pure MSE is that the learning is gonna be more stable? (confirm this too).
- **Discretized head (softmax over each actuator's bins):** Problem with both the above approaches is that.. if the training data saw 70 +1s and 40 -1s then the output is gonna get averaged to +0.4. This might be wrong, The training data is trying to convey that there are multiple ways to achieve the successful outcome but the policy is learning to average these individual policies. Instead, one way to handle this is to discretize each actuators' output into 256 bins (say) and then output a LLM style prob vector of size 256 for each actuator on every turn. This enables the robot policy to learn multiple trajectories to achieve success without averaging the policies.
- **Diffusion / flow head:** Problem with the Softmax policy mentioned above is that, say, bin 145 and 146 are different action classes from model's perspective.. but realistically they are very similar actions. This is not a problem with LLMs because 145 and 146 might actually be different classes semantically. Diffusion policies iteratively denoise the output to predict the specific trajectory. They have the ability to learn multiple trajectories.

---

## RL on real robots, and sim to real

### Instead of imitation learning, the robot learns by trial and error (RL). What are the concrete reasons RL on a real robot arm is much harder?

Real world training is time taking, hard to reset, expensive, hard to scale.

### Real data is expensive, so you train in a physics simulator where data is free. It works perfectly in sim and fails on the real robot. Why? Besides "make the sim more realistic," what's a cheaper fix?

This is because the real world has friction, the actuators are imperfect, the commands might be jumpy, hardware might be unreliable at times etc., As we saw before, these little imperfections compound quickly resulting in failed tasks. **Domain randomization**: Add random noise to the sim to make policy more robust thereby giving every simulated episode different world settings.

- **Visual:** random textures, colors, lighting, camera position. Tobin et al. (2017) used deliberately strange colors and textures.
- **Physical:** random object mass, friction, motor strength, time delays.

**Levine et al. (2016)**: This paper trained one CNN that maps raw pixels directly to motor torques.

**OpenAI Dactyl (2019)** trained a robot hand to rotate a cube, entirely in simulation, using heavy physics randomization. In each episode the physics was different, so the policy learned to notice something like "this cube is heavy and slippery" from the past few seconds and adjust.

---

## Robot policy learning approaches

There are various ways to achieve a learnt robot policy. Robot policies are mainly **DRL** with continuous actions.

### Algorithm families

- **Policy Gradient:** Output actions are generally discretized and trained with the policy output and an advantage. General algorithms are **REINFORCE**, **PPO**, **GRPO**. PPO is an **on-policy** algorithm so each update needs a fresh rollout and the older policy gets thrown away.
- **Q-learning family:** Uses DQN variants. There is no policy network. Instead the robot learns value network. **QT-Opt** is a DQN variant for robot learning. DQN family of algorithms are **off-policy**. This means you can train the network on older data (**replay buffer**).
- **Actor Critic:** DDPG and SAC train a second network (the "actor") whose only job is to output the action that maximizes Q. This is called actor-critic.

### How does QT-Opt work with continuous action? How would you find the action with the highest Q?

*Personal fav.* With CartPole you compute Q for both actions (forward, backward) and pick the larger. With 7 continuous numbers there are infinitely many actions to check. QT-Opt uses **CEM (Cross-Entropy Method)**, which is a guess and refine search:
- Sample 64 random action vectors.
- Score each one with Q(s, a).
- Keep the top 6.
- Fit a Gaussian (mean and spread) to those 6 and sample 64 new actions near them.
- Repeat 2–3 times and use the best action found.

The search moves toward the region where Q is high. It's a few dozen forward passes through Q, with no gradients involved. This approach is kind of similar to GRPO but with value network.

### Model-free vs model-based

From David Silver's lectures, there are 2 kinds of approaches to RL:

- **Model-free learning:** learn the policy only.
- **Model-based learning:** learn the policy *and* a model of the environment.

Both are after the same end product, a policy. Acting well requires some grasp of how the environment behaves, and the two differ in whether that knowledge is made "explicit". Model-free leaves it implicit inside the policy. Model-based learns the env dynamics separately and reaches the policy through that crutch. Examples of model-based are **MuZero**, **Dreamer**, **JEPA** and other world models.

| Component | Mapping | At decision time |
| --- | --- | --- |
| **Policy** | observation → action, directly | See, then act. One forward pass, no thinking ahead. |
| **World model** | (state, action) → next state, i.e. "what happens if I do this?" | Imagine several possible action sequences, predict where each leads, pick the best. That's **planning**. |

---

## Where does the data come from?

Another important question to think about is, where does the data come from?

- Either through imitation (BC we spoke about in section 1)
- Robot's own attempts through Reinforcement Learning.
- Learning in Sim and applying it in real.
- Internet data (text, video, images)

### Current classes of algorithms, by what they learn

Some algorithms learn just the policy. Some learn the concepts, intuition and physics through pretraining with vision and language data. Some learn the dynamics and then the policy, picking up the world's physics by watching videos.

- **Imitation learning** = policy + human demos. (BC, DAgger, ACT, Diffusion Policy)
- **VLAs** = policy + human demos + internet knowledge inherited from a pretrained vision-language model. The robot already knows what a "cup" or "the red one" is before seeing a single demo. (RT-2, OpenVLA, π0, π0.5)
- **World models / JEPA** = learn how the world changes, mostly from video, then plan inside it. (Dreamer, V-JEPA 2, DINO-WM)

---

## ACT

ACT, also called Action Chunking Transformer, predicts the next n-actions (say 100 from now) chunk instead of one action at a time.

Architecturally it is **not** a Vision Transformer. It is a **CVAE** (conditional variational autoencoder): a **ResNet18** backbone embeds each camera's image independently, those features feed a **transformer encoder-decoder**, and the decoder's learned positional queries (one per timestep in the chunk) cross-attend to the observation to emit the whole chunk in one pass. Trained with **L1 loss** on the actions plus a KL term on the latent.

Inputs are the camera images and joint positions (proprioception). The original ACT has **no language conditioning**, it is single-task. CVAE encoder embeds the expert's action sequence, but only at training time, compressing it into a style variable `z` that captures the demo's style. That `z` is what lets ACT handle multimodal demos. At test time `z` is set to zero (the prior mean), so the policy is deterministic given the observation.

Below is an example snippet of the trajectory.

| Inference at step | Predicts actions for steps |
| --- | --- |
| 1 | 1, 2, 3, **4** |
| 2 | 2, 3, **4**, 5 |
| 3 | 3, **4**, 5, 6 |
| 4 | **4**, 5, 6, 7 |

### Ensembling the overlapping predictions

You could then choose to do a weighted avg (with recency bias) of the predicted actions, or just execute action ensemble (1, 2) and then smooth action 3 & 4 etc.,

Note: In the above example, values of action at timestep 4 are averaged across all the other predictions.

In fact, I think you could do a weighted avg during train time and then do ensemble inference during test time.

### What problems does ACT fix?
- **It's better than predicting 1 action at a time** since you'd have a smoother trajectory because of averaging.
- **It's better than executing one long chunk from a single prediction** (say one 500-step chunk) because that'd lead to compounding errors (see above). The problem with one 500-step chunk was that executing each action puts you in a slightly stranger state. With ACT, we technically have 100 predictions for an action at a specific timestep. This smooths out the action value and puts you on a fairly stable trajectory.
- **A chunk commits to a single plan.** If a robot moves left, it'll continue to move left through the whole chunk. Rather, a chunk-size-1 robot's predictions are broken. action 1 says left, action 2 says right, etc.,
- *Personal fav.* When a human teleops a robot, human behavior naturally lags and is often jumpy. Say a normal teleop is 50 frames per sec. Imagine a human lags at the cup handle for 2 secs. This is 100 frames of training data with "no action". A chunk-size-1 ACT algorithm would undesirably learn to freeze for such input image because the teleop data for those 2 secs contains 100 frames of no action. With a chunk-size-100 ACT, we'd have 99 frames of "no action" and 100th frame would have a legit action. After 50 predictions; we'd have 50 frames of "no action" with following 50 frames of legit actions. Also, these future actions get smoothed as more and more predictions are made. Note: This avoids the freezing but the robot still learns undesirable human behaviors like freezing for 2 seconds.

## VAE

VAE is a successor of AE but solves the problem of efficiently distributing and packing the points in the latent space. Contrary to AE, it also enables the decoder to not remember a specific point to reconstruct an image. It rather enables the decoder to recognize a "region" to map an image to.

Now the KL loss, encoder's gaussian params output, reparametrization trick, and gaussian sampling the decoder input in latent space are all a series of tricks to enable the above behavior.

### The two losses

We mainly enforce 2 kinds of losses to enforce this behavior. Below is the intuition for each:

- **Reconstruction loss:** Helps distribute the class representations in the latent space far apart (to distinguish better) while trying to fully reconstruct the given image from the latent representation. It also actively tries to minimize the standard dev corresponding to each class. *Important: This loss has a standing incentive to spread centers arbitrarily far apart, because more separation means more distinguishability.*
- **KL loss:** KL loss helps efficiently pack the representations in the latent space with fewer gaps. In VAE, KL loss forces the representations to have the mean near the origin. It also penalizes if the variance becomes 0. This creates a tug of war between KL loss and reconstruction loss but enables the variance to be just enough to enforce variability in generating the class.

### Math

`VAE Loss = reconstruction + β · KL`

- Reconstruction loss is straight forward: `L_recon = ‖x − D(z)‖² = MSE(generated image - input image)`
- KL has a closed form for diagonal gaussians, summed over the latent dimensions `j`:
    ```
    KL( E(z|x) ‖ N(0,I) ) = 0.5 · Σ_j ( μ_j² + σ_j² − log σ_j² − 1 )
    ```
- `μ_j²` is a **spring pulling every blob's center to the origin**. `μ=5` costs 12.5, `μ=0` costs 0. This stops encoder from scattering classes off to `(-200, 8000)`, so the occupied region stays connected with no faraway fragmented islands.
- `σ_j² − log σ_j² − 1` **forces the width toward exactly 1**, punishing both directions.

| `σ` | per-dim KL (at `μ=0`) | why |
| --- | --- | --- |
| 1.0 | 0 | the target |
| 0.1 (too sharp) | 1.81 | `−log σ²` climbing |
| 0.01 | 4.10 | climbing harder |
| → 0 | **→ ∞** | infinite wall |
| 3.0 (too fuzzy) | 2.90 | `σ²` dominates |

### Encoder / decoder boundary

The encoder emits **two** vectors of latent size (32 in ACT), mean & log-variance.

```python
mu, logvar = encoder(x)        # each (batch, 32)
std = torch.exp(0.5 * logvar)  # always positive by construction
```

**Reparametrization trick:** Now sampling `z = torch.normal(mu, std)` kills the gradient, because "drawing a random number" has no useful derivative w.r.t. `mu`. So we pull the randomness out and make `z` plain arithmetic:

```python
eps = torch.randn_like(std)   # drawn OUTSIDE the computational graph, a constant
z = mu + std * eps            # one multiply, one add
```

### Why sampling `N(0,I)` actually produces an output?

A random `z` was never in the training data, so why isn't it a covariate shift problem for the decoder?

Because KL is charged **per example and averaged**, not on the aggregate. So each individual blob is a modest off-center bump that bought its position, but take 10,000 of them with centers mildly scattered around the origin and width ≈ 1 and their **union** looks like `N(0,I)`. The prior is matched by the *mixture*, not by any single blob.

So when you sample `z ~ N(0,I)` you are drawing from (approximately) the exact distribution the decoder was trained on. There is no covariate shift, not by luck but because KL **engineered the training-time input distribution to equal the test-time sampling distribution**.

A `z` you have genuinely never seen, sitting between two class blobs, decodes to something in between, because the blobs overlap and the decoder is smooth. That is latent interpolation, and it only works because the gaps were removed.

### Economics of latent space in VAE

Context: A nat is a unit of information, like a bit but using natural log instead of log base 2. One nat ≈ 1.44 bits. KL comes out in nats when you use ln.

KL loss essentially charges for the real estate in latent space. This is very similar to PCA & finding eigenvalues. (More than an analogy, in fact: a *linear* VAE with a gaussian likelihood provably recovers probabilistic PCA.) Basically the encoder asks "what do I tell the decoder so it can redraw this particular image completely?". Latent space then captures features that drive "lot of pixels" in the output. If you are training VAE to generate cat pictures, things like "color of fur", "size of the cat", "pose of the cat" etc., are high information signals that drive a lot of features in the generated pictures. VAE pays for these features because the pay-off is high per-nat (these specific features play a huge part in the reconstructed image). It wouldn't try to remember the features like "exact position of whiskers of cat in 40th image" because the pay-off is very low. Therefore it won't try to assign a real estate in the latent space. It'll let the decoder figure these finer details. Additionally, this is the reason why VAE generated images are not crisp like the ones generated through adversarial learning.

(The pixel-wise `MSE` is the second reason for the blur: when several reconstructions are plausible, the loss-minimizing output is their average, and averaging sharp alternatives gives a soft image. GANs avoid this because the discriminator penalizes "looks averaged" directly.)

**High payoff-per-nat (bought first):**

| Factor | Why it's cheap | Why it pays |
| --- | --- | --- |
| Body pose / orientation | a few numbers | wrong pose = every pixel wrong |
| Overall brightness | ~1 nat | shifts the whole image |
| Base fur color | ~2 nats | sets thousands of pixels |
| Background color | ~1 nat | sets most of the frame |
| Rough size / distance | ~1 nat | scales everything |

**Low payoff-per-nat (dropped):**

| Factor | Why it's expensive | Why it doesn't pay |
| --- | --- | --- |
| Individual whisker positions | ~20 independent values, each unpredictable | a few dozen pixels |
| Exact fur strand directions | essentially random, huge entropy | sub-pixel texture |
| Nail shapes | needs its own dedicated bits | tiny region |
| Precise pupil dilation | unpredictable from anything else | small, low-contrast |

Why KL *is* information, at the two extremes:

| Encoder output | KL | What `z` tells you about the input |
| --- | --- | --- |
| `mu=0, σ=1` (identical to prior) | 0 | Nothing. You could have drawn this `z` without ever seeing the image. |
| `mu=5, σ=0.01` (tight, far off-center) | large | A lot. This `z` pins down which input it came from. |

A dimension that never pays for itself gets switched off entirely: the encoder sets `mu=0, σ=1` for it, pays zero KL thereby carries zero information. So a 32-dim latent often has only 10-15 genuinely active dimensions, and you can measure which by logging per-dimension KL.

`β` is the dial on all of this. Raise it for a smoother, more disentangled space and blurrier output; lower it for sharper reconstructions, a **holier space** and worse sampling.

### cVAE, and ACT

The **C** is conditional: both encoder and decoder get an extra input `c`.

- Encoder: `(x, c) → z`
- Decoder: `(z, c) → x`

With cats, set `c = "black"`. Now `z` doesn't need to encode "color aspect" since `c` already states it, so `z` carries only **what's left over**: pose, fur texture, lighting.

Mapped onto [ACT](#act):

| cVAE | ACT |
| --- | --- |
| `c`, the condition | the observation (camera images + joint positions) |
| `z`, the leftover style | **which of the many valid ways** the human did this demo |
| decoder output | the chunk of 100 actions |

#### At training time:

```
(joints + the actual expert action chunk) -> E -> z
(image + joints, z)                      -> D -> predicted action chunk
```

- Compute predicted chunk's loss against the expert next chunk.
- `z` conveys the *nuance* (is it left leaning, is it in a slippery domain etc.,) of the current demo. Given z, the policy (Decoder) no longer has to hedge across all the valid variants. The *"hedge"* here is the same averaging problem from [Various outputs and losses for robot policy](#various-outputs-and-losses-for-robot-policy), solved with a latent instead of binning or diffusion.

#### At test time:

- There is no expert chunk, so there is no `z` to compute. We set `z = 0`, the mean of the prior, meaning "give me the most typical style for this observation". Note this makes ACT **deterministic** at inference: it spends the cVAE purely to fix training-time averaging, then discards the diversity.

#### Why use only Decoder and not Encoder during inference? What's the intuition?

- Decoder is the policy/student that learns to predict the right next actions. Encoder is more like a teacher that teaches the Decoder the nuances of the observations like "this particular demo is the left-handed variant". Essentially Encoder infers the modality (see [Glossary](#glossary)) of the trajectory making it easier for Decoder to learn the policy. This also teaches Decoder to not avg different modalities, i.e., "the average of two valid options is not valid".
- Decoder relies on Encoder to give it the modality. Decoder does not learn it by itself.
- **Important takeaway:** *This brings us to one of cVAE/ACT's key drawbacks. If you observe, we set the z to 0 during test time. This is because there is no Encoder to give us the z. This means, we are essentially setting a default modality of 0 thereby giving up test-time mode selection. Other approaches solve this by conditioning it on an observation (like language command), sampling z etc., This is one of the key problems solved along the lineage of the robot policy learning.*

#### Why do we pass image to Decoder but not to Encoder?

- So that encoder doesn't spend much of its time trying to encode the image in the latent space
- We do set joint observations to both E & D because that's the only thing we want the z to encode or capture the nuance of.
- Faster inference during train time.
- Onus is on Decoder to capture the nuances in the images since z is not going to capture it.

---

## Observations

### LLM mid-training is imitation learning

*Personal fav.*

**Premise:** The key difference between IL and RL is that RL needs to figure out the signal of the trajectory by itself, IL already knows the signal of the trajectory (generally positive) so we just use the trajectory to finetune the model. This is a key distinction to remember.

**Pipeline shift:** Prior to 2025, LLM training involved pre-training + post-training. Now it's pre + mid + post. Post-training is mainly SFT + RLHF/RLVR. In post-training, the model simulates its own trajectories and finds its own signals in the environment.

**Why mid-training:**

- Researchers found that waiting until the very end to teach a model how to think acts like a shallow veneer. If you want a model to be deeply good at math or logic, why not give it those priors early on in the training?
- Pre-2025, you make a pre-trained model run its own simulations in an environment to find its own signals which are used to train the model. It's hard to have a naive model do these simulations without wasting a lot of compute!

**What is mid-training:** Now that we have some of these successful trajectories (from user data, or our own agentic task sandbox simulations), we could train the model on these successful trajectories through imitation learning. This is mid-training. This gives models useful priors like how to reason, how to use tools etc., before reaching SFT. In this stage, models are given thinking traces of successful complex math proofs, complex tool calling, complex reasoning traces etc.,

**Intuitively:**

- **Pre-training** incorporates the mammalian knowledge. This knowledge on its own is raw and unproductive.
- **Mid-training** incorporates useful skills. This knowledge teaches the model the relevant structures of the world that it ought to operate in.
- **Post-training** helps the model put these useful skills at play, run its own experiments and learn. This phase teaches the model to experience the nuances of the skills it learned in mid-training. A model cut off after mid-training would fail often at tasks and would often not try different approaches to the same problem.

---

## Glossary

- **Multi-modality:** It is the nuance of the trajectory/observation. Mathematically, each state can have multiple right actions. This is generally multiple peaks. Real life examples are, "Left-leaning", "fast moving" etc., Mind you, this is not the "stylistic flavor" but rather the actual >1 modes.

---

## Questions

- Influence of data availability on choosing to use various model types.. for example, Diffusion is taking off in robot learning (vs using transformers) because of the data scarcity. Transformer architectures need lot of training data which is scarce in robot domain. (Validate invalidate this hypothesis).
