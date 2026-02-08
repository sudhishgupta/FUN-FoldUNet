import torch
import numpy as np


class NoiseScheduler:
    """
    Noise scheduler supporting 'linear' and 'cosine' schedules.

    Usage:
        scheduler = NoiseScheduler(
            num_timesteps=1000,
            beta_start=0.0001,      # only used for linear schedule
            beta_end=0.02,          # only used for linear schedule
            schedule='cosine',      # or 'linear'
            s=0.008,                # small offset for cosine schedule
            device='cuda'
        )
    """
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=2e-2,
                 schedule='linear', s=0.008, device='cuda'):
        self.device = torch.device(device)
        self.num_timesteps = int(num_timesteps)
        self.schedule = schedule.lower()
        self.s = float(s)

        if self.schedule == 'linear':
            # original linear schedule (kept for compatibility)
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
        elif self.schedule == 'cosine':
            betas = self._cosine_betas(self.num_timesteps, s=self.s, device=self.device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Choose 'linear' or 'cosine'.")

        # clip betas into (0,1)
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)  # \bar\alpha_t
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)

    def _cosine_betas(self, timesteps, s=0.008, device='cpu'):
        """
        Cosine schedule (implementation from Nichol & Dhariwal / common libs).
        Returns a torch.tensor of shape [timesteps] on specified device.
        """
        steps = timesteps + 1
        # Use numpy for stable float ops, then convert to torch
        t = np.linspace(0, timesteps, steps, dtype=np.float64)
        f = np.cos(((t / timesteps) + s) / (1 + s) * (np.pi / 2.0)) ** 2
        alpha_cumprod = f / f[0]
        # betas are 1 - alpha_cumprod[t] / alpha_cumprod[t-1]
        betas = 1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = np.clip(betas, a_min=0.0, a_max=0.999)
        return torch.tensor(betas, dtype=torch.float32, device=device)

    # FORWARD DIFFUSION PROCESS
    def add_noise(self, original, noise, t):
        t = t.to(self.device)
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        xt = sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
        return xt

    # REVERSE DIFFUSION PROCESS
    def sample_prev_timestep(self, xt, noise_pred, t):
        t = torch.tensor(t, device=self.device) if not torch.is_tensor(t) else t.to(self.device)
        # predict x0
        x0_pred = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / (self.sqrt_alpha_cum_prod[t] + 1e-12)
        # compute mean
        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t] + 1e-12))
        mean = mean / torch.sqrt(self.alphas[t] + 1e-12)

        if int(t.item()) == 0:
            return mean, x0_pred
        else:
            # variance according to DDPM paper
            variance = (1.0 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = torch.sqrt(variance + 1e-12)
            z = torch.randn(xt.shape, device=xt.device)
            sample = mean + sigma * z
            return sample, x0_pred
