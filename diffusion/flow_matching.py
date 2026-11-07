from pathlib import Path

import torch
from torch.nn.modules import Module, Linear, Sequential, GELU
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam

EPOCHS = 3000
DIFFUSION_STEPS = 10
TRAIN_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 256
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "flow_match.pt"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOT_EVERY = 250


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


def train(means, stds):
    batch_means = means.expand(TRAIN_BATCH_SIZE, 4)
    batch_stds = stds.expand(TRAIN_BATCH_SIZE, 4)

    model = FlowMatch()
    optim = Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

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

        if e % PLOT_EVERY == 0:
            with torch.no_grad():
                samples = inference(model)
            plot(
                samples,
                path=PLOTS_DIR / f"epoch_{e:05d}.png",
                title=f"epoch {e}, loss {loss.item():.3f}",
            )

    return model


def inference(model):
    # model = FlowMatch()
    xt = torch.randn(INFERENCE_BATCH_SIZE, 4)
    for t in range(0, DIFFUSION_STEPS):
        append = torch.full((INFERENCE_BATCH_SIZE, 1), float(t) / DIFFUSION_STEPS)
        x = torch.cat((xt, append), dim=1)
        v = model(x)
        xt = v * (float(1) / DIFFUSION_STEPS) + xt
    return xt


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


if __name__ == "__main__":
    # train()
    # print(torch.normal(mean=means, std=stds))
    # model = FlowMatch()
    means = torch.tensor([8.1, 5.0, 61.3, 22.8])
    stds = torch.tensor([2.1, 1.3, 7.7, 2.8])
    model = train(means=means, stds=stds)
    save_model(model)
    plot(inference(model))
    # print(model(torch.randn(8, 5)))
