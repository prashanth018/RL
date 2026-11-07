import torch
from torch.nn.modules import Module, Linear, Sequential, GELU
import torch.nn as nn
from torch.optim import Adam


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


def train():
    means = torch.tensor([8.1, 5.0, 61.3, 22.8])
    stds = torch.tensor([2.1, 1.3, 7.7, 2.8])
    batch_size = 8
    batch_means = means.expand(batch_size, 4)
    batch_stds = stds.expand(batch_size, 4)
    epochs = 10

    model = FlowMatch()
    optim = Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    for e in range(epochs):
        dest = torch.normal(mean=batch_means, std=batch_stds)
        source = torch.rand_like(dest)
        t = torch.rand(batch_size, 1)
        xt = (1 - t) * source + t * dest
        pred = model(torch.concat((xt, t), dim=0))
        target = dest - source
        loss = mse_loss(input=pred, target=target)
        optim.zero_grad()
        loss.backward()
        optim.step()


def inference():
    pass


if __name__ == "__main__":
    # train()
    # print(torch.normal(mean=means, std=stds))
    model = FlowMatch()
    print(model(torch.rand(8, 5)))
