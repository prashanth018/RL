from pathlib import Path

import torch
from torch.nn.modules import Module, Linear, Sequential, GELU
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam

EPOCHS = 9000
DIFFUSION_STEPS = 10
TRAIN_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 256
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "flow_match.pt"
PLOTS_DIR = Path(__file__).parent / "plots"
LOG_EVERY = 100


class FlowMatch(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(in_features=5, out_features=64),
            GELU(),
            Linear(in_features=64, out_features=64),
            GELU(),
            Linear(in_features=64, out_features=64),
            GELU(),
            Linear(in_features=64, out_features=4),
        )

    def forward(self, x):
        return self.net(x)


def sample_mixture(n, means, stds, weights):
    idx = torch.multinomial(weights, n, replacement=True)
    return torch.normal(mean=means[idx], std=stds[idx]).unsqueeze(1)


def train(means, stds):
    batch_means = means.expand(TRAIN_BATCH_SIZE, 4)
    batch_stds = stds.expand(TRAIN_BATCH_SIZE, 4)

    model = FlowMatch()
    optim = Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    losses = []
    running_loss = 0.0

    for e in range(EPOCHS):
        dest = torch.normal(mean=batch_means, std=batch_stds)
        source = torch.randn_like(dest)
        t = torch.rand(TRAIN_BATCH_SIZE, 1)
        xt = (1 - t) * source + t * dest
        pred = model(torch.concat((xt, t), dim=1))
        target = dest - source
        loss = mse_loss(input=pred, target=target)
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        if (e + 1) % LOG_EVERY == 0:
            losses.append((e + 1, running_loss / LOG_EVERY))
            running_loss = 0.0

    return model, losses


def inference_trajectory(model, n):
    xt = torch.randn(n, 4)
    trajectory = [xt]
    for t in range(0, DIFFUSION_STEPS):
        append = torch.full((n, 1), float(t) / DIFFUSION_STEPS)
        x = torch.cat((xt, append), dim=1)
        v = model(x)
        xt = v * (float(1) / DIFFUSION_STEPS) + xt
        trajectory.append(xt)
    return torch.stack(trajectory)


def inference(model):
    return inference_trajectory(model, INFERENCE_BATCH_SIZE)[-1]


def save_model(model, path=CHECKPOINT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path=CHECKPOINT_PATH):
    model = FlowMatch()
    model.load_state_dict(torch.load(path))
    return model


def plot(x, path=None, title=None):
    x = x.detach().cpu().numpy()
    dims = x.shape[1]
    fig, axes = plt.subplots(1, dims, figsize=(4 * dims, 3), squeeze=False)
    for d, ax in enumerate(axes[0]):
        ax.hist(x[:, d], bins=50)
        ax.set_title(f"dim {d}: mean={x[:, d].mean():.1f}, std={x[:, d].std():.1f}")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def plot_loss(losses, path):
    steps, values = zip(*losses)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Loss (mean of last {LOG_EVERY} steps)")
    ax.set_title("Flow matching training loss")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_trajectory(trajectory, means, stds, path):
    trajectory = trajectory.detach().cpu().numpy()
    ts = [i / DIFFUSION_STEPS for i in range(DIFFUSION_STEPS + 1)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    dims = trajectory.shape[1]
    fig, axes = plt.subplots(1, dims, figsize=(4 * dims, 3.5), squeeze=False)
    for d, ax in enumerate(axes[0]):
        m, s = means[d].item(), stds[d].item()
        ax.axhspan(m - s, m + s, color="#9a9a96", alpha=0.18, linewidth=0, label="target μ ± σ")
        ax.axhline(m, color="#9a9a96", linestyle="--", linewidth=1)
        ax.plot(ts, trajectory[:, d], color=colors[d], linewidth=2, marker="o", markersize=5)
        ax.annotate(f"{trajectory[-1, d]:.1f}", (ts[-1], trajectory[-1, d]), textcoords="offset points", xytext=(-6, 8), ha="right", fontsize=9)
        ax.set_title(f"dim {d}: μ={m:.1f}, σ={s:.1f}")
        ax.set_xlabel("t  (0 = noise, 1 = sample)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].set_ylabel("value")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("One sample's path from noise to the target distribution")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    # train()
    # print(torch.normal(mean=means, std=stds))
    # model = FlowMatch()
    means = torch.tensor([8.1, 5.0, 61.3, 22.8])
    stds = torch.tensor([2.1, 1.3, 7.7, 2.8])
    model, losses = train(means=means, stds=stds)
    save_model(model)
    with torch.no_grad():
        samples = inference(model)
    plot(
        samples,
        path=PLOTS_DIR / f"epoch_{EPOCHS:05d}.png",
        title=f"epoch {EPOCHS}, loss {losses[-1][1]:.3f}",
    )
    plot_loss(losses, PLOTS_DIR / "loss_curve.png")
    with torch.no_grad():
        trajectory = inference_trajectory(model, 1)[:, 0]
    plot_trajectory(trajectory, means, stds, PLOTS_DIR / "trajectory.png")
    # print(model(torch.randn(8, 5)))
