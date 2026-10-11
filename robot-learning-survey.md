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
8. [Questions](#questions)

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

## Questions

- Influence of data availability on choosing to use various model types.. for example, Diffusion is taking off in robot learning (vs using transformers) because of the data scarcity. Transformer architectures need lot of training data which is scarce in robot domain. (Validate invalidate this hypothesis).
