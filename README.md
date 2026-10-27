# RL

- [Robot Learning Survey](robot-learning-survey.md): behavior cloning, covariate shift, ACT, VAE/cVAE, sim-to-real, policy learning approaches
- [ACT on ALOHA transfer cube](act/README.md): reproduced lerobot's reference (82% vs 83%), then a chunk size + temporal ensembling ablation that reaches 92%
- [Reinforcement Learning Debugging](rl-debugging.md) [WIP]:  Guide on RL debugging, isolating components, bisecting the problem, cons of adaptive reward scaling, pros of chasing anomalies, long horizon training stability.
- [LeRobot](lerobot.md): architecture and data layout, `delta_timestamps`, teleop data collection, FK/IK/Diff-IK, SAC
- [DQN with replay buffer](fa/README.md): 6 training runs on CartPole-v1, catastrophic forgetting, DDQN
- [DQN vs Q-learning](fa/DQN_vs_QLearning.md): what DQN solves (sample correlation, moving targets), the maximization bias it introduces, and how Double DQN addresses it
