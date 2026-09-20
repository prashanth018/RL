import torch

from dqn import DQNAgent, GAMMA, NUM_EPISODES

MODEL_FOLDER = "ddqn_replay_buffer"


class DDQNAgent(DQNAgent):
    def __init__(self):
        super().__init__(model_folder=MODEL_FOLDER)

    def compute_target(self, batch):
        _, _, rewards, next_states, dones = zip(*batch)

        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            # decouple action selection (online network) from action evaluation (target network)
            #   a* = argmax_a Q(s', a; updateQN)       <- online network selects
            #   y  = r + gamma * Q(s', a*; targetQN)   <- target network evaluates
            optimal_actions = self.updateQN(next_states).argmax(dim=1, keepdim=True)
            targets = rewards + (1 - dones) * GAMMA * self.targetQN(next_states).gather(
                1, optimal_actions
            ).squeeze(1)

        return targets


if __name__ == "__main__":
    sim = DDQNAgent()
    # for ep in range(NUM_EPISODES):
    #     print(f"Running episode {ep}")
    #     sim.episode()
    # sim.plot_stats()
    sim.visualize()
