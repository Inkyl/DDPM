import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.pos = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        x = self.fc1(torch.cat([x, t], dim=1))
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class Diffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, T, *args, **kwargs):
        super(Diffusion, self).__init__()
        self.model = MLP(input_dim * 2, hidden_dim, output_dim)
        self.T = T
        self.device = kwargs['device']
        self.betas = torch.linspace(
            0.0001, 0.02, self.T, dtype=torch.float32, device=self.device
        )
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.betas_bar = 1 - self.alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
        self.sqrt_recip_alphas_bar = torch.sqrt(1 / self.alphas_bar)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_bar_prev = torch.cat([torch.ones(1, device=kwargs["device"]), self.alphas_bar[:-1]])
        self.one_minus_alphas_bar_prev = 1.0 - self.alphas_bar_prev
        self.sqrt_alphas_bar_prev = torch.sqrt(self.alphas_bar_prev)
        self.recip_one_minus_alphas_bar = 1.0 / self.one_minus_alphas_bar

    # 加噪过程
    def q_sample(self, x0, t, noise=None):
        if noise == None:
            noise = torch.randn_like(x0)
        return self.sqrt_alphas_bar[t] * x0 + self.sqrt_one_minus_alphas_bar[t] * noise

    # 采样过程

    # 用xt预测x0
    def get_x0_from_xt(self, xt, t, pred_noise):
        return self.sqrt_recip_alphas_bar[t] * xt - self.sqrt_one_minus_alphas_bar[t] * pred_noise

    def q_posterior(self, x0, xt, t):
        posterior_mean = (self.sqrt_alphas[t] * self.one_minus_alphas_bar_prev[t] * xt + self.sqrt_alphas_bar_prev[t] *
                          self.betas[t] * x0) * self.recip_one_minus_alphas_bar[t]
        posterior_variance = self.betas[t] * self.one_minus_alphas_bar_prev[t] * self.recip_one_minus_alphas_bar[t]
        posterior_log_variance = torch.log(posterior_variance)

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, xt, t):
        pred_noise = self.model(xt, t)
        x_recon = self.get_x0_from_xt(xt, t, pred_noise)
        x_recon.clamp_(-1, 1)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, xt, t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, xt, t, noise=None):
        b = xt[0]
        if noise == None:
            noise = torch.randn_like(xt)
        posterior_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(xt, t)
        # 得到的是b*其他维度都为1的shape
        # 当t不为0的时候 这个mask为1，否则为0 目的是t=0的时候不引入噪声
        mask = (1 - (t == 0)).float().reshape(b, *((1,) * (len(xt) - 1)))
        return posterior_mean + mask * torch.exp(0.5 * posterior_log_variance) * noise

    def p_sample_loop(self, device, shape):
        batch_size = shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)
        for i in range(self.T - 1, -1, -1):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x
