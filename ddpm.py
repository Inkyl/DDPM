import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from unet import NaiveUnet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


class Diffusion(nn.Module):
    def __init__(self, in_channels, out_channels, betas, T, device):
        super(Diffusion, self).__init__()
        self.model = NaiveUnet(in_channels, out_channels, n_feat=128)
        self.T = T
        self.device = device
        for k, v in ddpm_schedules(*betas, T).items():
            self.register_buffer(k, v)
        self.loss_fn = nn.MSELoss()

    def forward(self, x0):
        t = torch.randint(1, self.T + 1, size=(x.shape[0],), device=device)
        noise = torch.randn_like(x0, device=device)
        xt = self.sqrt_alphas_bar[t, None, None, None] * x0 + self.sqrt_one_minus_alphas_bar[
            t, None, None, None] * noise
        return self.loss_fn(noise, self.model(xt, t / self.T))

    def sample(self, n_sample, shape, device):
        xt = torch.randn(n_sample, *shape).to(device)
        for t in range(self.T, 0, -1):
            z = torch.randn(n_sample, *shape).to(device) if t > 1 else 0
            xt = self.sqrt_recip_alphas[t] * \
                 (xt - self.betas[t] /
                  self.sqrt_one_minus_alphas_bar[t] *
                  self.model(xt, torch.tensor(t / self.T).to(device).repeat(n_sample, 1))) + \
                 self.sqrt_betas[t] * z
        return xt


def ddpm_schedules(beta1: float, beta2: float, T: int):
    betas = torch.linspace(beta1, beta2, T + 1, dtype=torch.float32)
    alphas = 1 - betas
    log_alphas = torch.log(alphas)
    alphas_bar = torch.cumsum(log_alphas, dim=0).exp()
    one_minus_alphas_bar = 1 - alphas_bar
    sqrt_alphas = torch.sqrt(alphas)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(one_minus_alphas_bar)
    sqrt_recip_alphas = 1 / sqrt_alphas
    sqrt_betas = torch.sqrt(betas)

    return {'betas': betas,
            'sqrt_alphas_bar': sqrt_alphas_bar,
            'sqrt_one_minus_alphas_bar': sqrt_one_minus_alphas_bar,
            'sqrt_recip_alphas': sqrt_recip_alphas,
            'sqrt_betas': sqrt_betas,
            }


if __name__ == '__main__':
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    ddpm = Diffusion(3, 3, (0.0001, 0.02), 1000, device)
    ddpm.to(device)
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "/mnt/data0/xuekang/workspace/datasets",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)
    n_epoch = 100
    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            os.makedirs('./contents', exist_ok=True)
            save_image(grid, f"./contents/ddpm_sample_cifar{i}.png")
            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_cifar.pth")
