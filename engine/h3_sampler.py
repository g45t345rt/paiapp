"""MiniMax H3 rectified-flow Euler scheduler (plain torch).

A self-contained port of Diffusers' ``MiniMaxH3Scheduler`` (Apache-2.0, MiniMax/HF
teams). Three things differ from a stock flow-match Euler:

1. The velocity is **data-ward**, so ``x0 = x_t + sigma * v`` (note the ``+``).
2. Timesteps are ``t = 1 - sigma`` in ``[0, 1]`` with ``t = 1`` meaning *clean*,
   which is what the DiT's AdaLN consumes directly.
3. The sigma grid is ``linspace(1, 0, n)`` pushed through the exponential shift,
   terminal ``0`` included, then de-duplicated.

H3 runs **two schedules per request**: ``shift = 12.0`` for video, ``3.0`` for
audio (:data:`VIDEO_SHIFT` / :data:`AUDIO_SHIFT`).
"""

import torch

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def shifted_sigmas(num_steps, shift):
    """``linspace(1, 0, num_steps)`` through ``s*sig/(1+(s-1)*sig)``, de-duplicated."""
    base = torch.linspace(1.0, 0.0, int(num_steps), dtype=torch.float32)
    sigmas = shift * base / (1 + (shift - 1) * base)
    return torch.unique_consecutive(sigmas)


class MiniMaxH3Scheduler:
    """Rectified-flow Euler scheduler (eta = 0) with an exponential sigma shift."""

    def __init__(self, shift=VIDEO_SHIFT):
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}.")
        self.shift = float(shift)
        self.sigmas = None
        self.timesteps = None
        self._step_index = None

    def set_timesteps(self, num_inference_steps, device=None, sigmas=None):
        """Build the sigma / timestep schedule.

        ``sigmas`` holds ``num_inference_steps`` values (terminal ``0`` included);
        ``timesteps = 1 - sigmas[:-1]`` drives ``num_inference_steps - 1`` model
        evaluations.
        """
        if sigmas is None:
            sigmas = shifted_sigmas(num_inference_steps, self.shift)
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
        self.sigmas = sigmas.to(device=device)
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self._step_index = None

    def scale_noise(self, sample, timestep, noise):
        """Forward process in the ``t`` convention: ``x_t = t*x0 + (1 - t)*noise``."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(self, model_output, timestep, sample):
        """One Euler step. Returns the next sample ``x_{t+1}``.

        ``denoised = x_t + sigma * v`` and ``next = r*x_t + (1 - r)*denoised`` with
        ``r = sigma_next / sigma``, evaluated in float32.
        """
        if self._step_index is None:
            self._step_index = int((self.timesteps == timestep).nonzero()[0].item())

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev = ratio * sample.to(compute_dtype) + (1.0 - ratio) * denoised.to(compute_dtype)
        self._step_index += 1
        return prev.to(dtype=sample.dtype)
