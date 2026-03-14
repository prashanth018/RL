import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import gymnasium as gym
import torch.optim as optim

"""
Env: Cart Pole
State: (4,)
    [position, velocity, angle, angular velocity]
Action: (1,) [0 is left; 1 is right]
"""

"""
Action Items:
- Implement ReplayBuffer
- Implement DQN; Init UpdateQNetwork, TargetQNetwork
- Initialize gym cart pole env and play a random episode (all right)
"""

GAMMA = 0.99
ALPHA = 0.01
EPSILON = 0.1
STATE_SIZE = 4
ACTION_SIZE = 2
BATCH_SIZE = 32
LR = 0.001
WEIGHT_TRANFER_CYCLES = 100


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            torch.tensor(state, dtype=torch.float32),
            action,
            reward,
            torch.tensor(next_state, dtype=torch.float32),
            done,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SIZE),
        )

    def forward(self, x):
        return self.net(x)


class DQNSim:
    def __init__(self):
        self.buffer = ReplayBuffer(1000)
        self.updateQN = DQN()
        self.targetQN = DQN()
        # sync the update and target DQNs
        self.targetQN.load_state_dict(self.updateQN.state_dict())
        self.optimizer = optim.RMSprop(self.updateQN.parameters(), lr=LR)
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.epsilon = EPSILON
        self.avg_loss_per_episode = []
        self.total_rewards_per_episode = []
        self.total_timesteps_per_episode = []
        self.total_steps = 0

    def epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.updateQN(state)
                return q_values.argmax().item()

    def compute_target(self, batch):
        _, _, rewards, next_states, dones = zip(*batch)

        next_states = torch.stack(next_states)  # [batch_size, state_size]
        rewards = torch.tensor(rewards)  # [batch_size, 1]
        dones = torch.tensor(dones)  # [batch_size, 1]

        with torch.no_grad():
            # if terminal state: target = reward
            # else: target = reward + gamma * max Q_target(s', a')
            targets = rewards + GAMMA * self.targetQN(next_states).max(dim=1).values * (
                1 - dones
            )  # [batch_size, 1]

        return targets

    def compute_prediction(self, batch):
        states, actions, _, _, _ = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions)

        predictions = self.updateQN(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        return predictions

    def plot_stats(self):
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.total_rewards_per_episode)
        axes[0].set_title("Total Reward per Episode")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")

        axes[1].plot(self.avg_loss_per_episode)
        axes[1].set_title("Avg Loss per Episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Loss")

        axes[2].plot(self.total_timesteps_per_episode)
        axes[2].set_title("Timesteps per Episode")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Timesteps")

        plt.tight_layout()
        plt.savefig("training_stats.png")
        plt.show()

    def episode(self):
        state, info = self.env.reset()
        episode_timesteps = 0
        episode_rewards = 0
        episode_loss = 0
        done = False
        terminated, truncated = False, False
        while not done:
            # get action
            action = self.epsilon_greedy(state)

            # take a step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # save to buffer
            self.buffer.push(state, action, reward, next_state, done)

            # update episodic reward & timesteps
            episode_timesteps += 1
            episode_rewards += reward

            # gradient update
            if self.buffer.__len__() >= BATCH_SIZE:
                batch = self.buffer.sample(BATCH_SIZE)
                target = self.compute_target(batch)
                prediction = self.compute_prediction(batch)

                loss = nn.MSELoss()(target, prediction)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update loss
                episode_loss += loss.item()

            self.total_steps += 1

            # weight transfer from updateQN to targetQN & save weights
            if self.total_steps % WEIGHT_TRANFER_CYCLES == 0:
                self.targetQN.load_state_dict(self.updateQN.state_dict())
                torch.save(
                    self.updateQN.state_dict(),
                    f"weights_timesteps{self.total_steps}.pth",
                )

            # decay epsilon
            self.epsilon = self.epsilon * 0.99

        self.total_timesteps_per_episode.append(episode_timesteps)
        self.total_rewards_per_episode.append(episode_rewards)
        if episode_timesteps > 0:
            self.avg_loss_per_episode.append(episode_loss / episode_timesteps)

        print(f"Terminated = {terminated}, Truncated={truncated}")


if __name__ == "__main__":
    sim = DQNSim()
    NUM_EPISODES = 5
    for ep in range(NUM_EPISODES):
        sim.episode()
    sim.plot_stats()
