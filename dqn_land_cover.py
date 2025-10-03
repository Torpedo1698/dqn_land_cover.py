import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# -----------------------------
# Environment: Land Cover Grid
# -----------------------------
class LandCoverEnv:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.target = [self.size - 1, self.size - 1]
        return self._get_state()

    def step(self, action):
        if action == 0:   # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2: # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        done = (self.agent_pos == self.target)
        reward = 1 if done else -0.01
        return self._get_state(), reward, done

    def _get_state(self):
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.target[0], self.target[1]] = 2
        return grid.flatten()

# -----------------------------
# Deep Q-Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# -----------------------------
# Training Function
# -----------------------------
def train_dqn(episodes=200):
    env = LandCoverEnv()
    state_dim = env.size * env.size
    action_dim = 4

    policy_net = DQN(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    gamma = 0.9
    epsilon = 0.3
    rewards_per_episode = []

    # Make output folder
    os.makedirs("output", exist_ok=True)

    for ep in range(episodes):
        state = torch.FloatTensor(env.reset())
        total_reward = 0

        for _ in range(50):
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.randint(0, action_dim-1)
            else:
                action = torch.argmax(policy_net(state)).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # Q-learning target
            target = reward + gamma * torch.max(policy_net(next_state)).item()
            output = policy_net(state)[action]

            loss = criterion(output, torch.tensor(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward
            if done:
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")

    # -----------------------------
    # Save outputs
    # -----------------------------
    # Save graph
    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("DQN Training on Land Cover Environment")
    plt.grid(True)
    plt.savefig("output/results.png")
    plt.show()  # display graph inline in Colab

    # Save training logs
    with open("output/training_logs.txt", "w") as f:
        f.write(str(rewards_per_episode))

    print("✅ Training complete. Outputs saved in 'output/' folder.")

# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    train_dqn()
