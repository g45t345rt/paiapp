"""Extra generation samplers and schedulers (plain torch/numpy).

This module ports the ComfyUI / k-diffusion sampler and scheduler families that
:mod:`engine.sampler` did not already implement, so the Generate panel offers the
same set the community workflows use.

Every sampler here shares one uniform call shape::

    sampler(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
            callback=None, generator=None, guider=None, ms=None, **options)

``model`` is the apply-model wrapper (as in :mod:`engine.sampler`), ``noise`` is
the already noise-scaled latent, and ``sigmas`` is the descending sigma tensor
whose last entry is ``0``. ``guider(model, x, sigma, ctx_pos, ctx_neg, cfg)``
returns the guided ``x0`` prediction for a step (default: plain CFG); the
``*_cfg_pp`` samplers instead read the conditional and unconditional predictions
separately, so they always use plain CFG.

The formulas are parameterisation-agnostic where possible, i.e. they consume the
model's ``x0`` prediction and the Karras sigma, exactly like the reference
implementations. A few families are, by construction, only meaningful for the
VP/epsilon schedule (``ddpm``); those are noted in their docstrings.

Everything depends on numpy + torch only. Brownian-tree correlated noise for the
SDE samplers is replaced by an uncorrelated noise sampler, which keeps those
samplers dependency-free while staying numerically valid.

The ``beta`` scheduler takes two shape parameters, ``alpha`` and ``beta``
(``0.5`` and ``0.7`` by default); higher values push more steps toward the corresponding end
of the schedule. They are exposed as ``BETA_ALPHA_DEFAULT`` /
``BETA_BETA_DEFAULT`` and can be overridden per run.
"""

BETA_ALPHA_DEFAULT = 0.5
BETA_BETA_DEFAULT = 0.7

import functools
import math

import numpy as np
import torch

try:  # scipy is already a runtime dependency; keep the import defensive.
    from scipy.special import betaincinv as _betaincinv
except Exception:  # pragma: no cover - scipy is listed in requirements.txt
    _betaincinv = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def append_dims(x, target_dims):
    """Append trailing singleton dims so ``x`` broadcasts against a target."""
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def to_d(x, sigma, denoised):
    """Karras ODE derivative from a denoiser (``x0``) output."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """``sigma_down`` to step to and ``sigma_up`` of noise to re-inject."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5,
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def make_noise_sampler(x, generator):
    """A ``noise_sampler(sigma, sigma_next)`` drawing ``randn_like(x)``.

    When ``generator`` is given the draw comes from it on the CPU and is moved to
    the latent's device/dtype, so a seeded run is reproducible.
    """
    if generator is None:
        return lambda sigma, sigma_next: torch.randn_like(x)

    def sampler(sigma, sigma_next):
        noise = torch.randn(
            x.shape, generator=generator, dtype=torch.float32, device="cpu"
        )
        return noise.to(device=x.device, dtype=x.dtype)

    return sampler


def _is_flow(ms):
    """True for the flow-matching samplings (Anima/Flux/Qwen/Z/Krea2/MOSS/H3)."""
    return ms is not None and hasattr(ms, "shift")


def _lambda(ms, sigma):
    """Half-logSNR ``log(alpha_t / sigma_t)`` as ComfyUI defines it per family."""
    sigma = torch.as_tensor(sigma)
    if _is_flow(ms):
        return sigma.clamp(1e-6, 1.0 - 1e-6).logit().neg()
    return sigma.clamp_min(1e-12).log().neg()


def _lambda_inv(ms, half_log_snr):
    if _is_flow(ms):
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()


def _offset_sigma_for_snr(sigmas, ms):
    """Clamp an initial sigma of ``1`` so the flow half-logSNR stays finite."""
    if _is_flow(ms) and len(sigmas) > 1 and float(sigmas[0]) >= 1.0:
        sigmas = sigmas.clone()
        sigmas[0] = 1.0 - 1e-4
    return sigmas


def _set_step(model, index, total):
    setter = getattr(model, "set_step", None)
    if setter is not None:
        setter(index, total)


def _plain_guider(model, x, sigma, ctx_pos, ctx_neg, cfg):
    cond = model.apply_model(x, sigma, ctx_pos)
    if math.isclose(float(cfg), 1.0):
        return cond
    uncond = model.apply_model(x, sigma, ctx_neg)
    return uncond + (cond - uncond) * cfg


def _guided_cfg(model, x, sigma, ctx_pos, ctx_neg, cfg):
    """Return ``(guided_x0, uncond_x0)`` for the CFG++ samplers."""
    cond = model.apply_model(x, sigma, ctx_pos)
    if math.isclose(float(cfg), 1.0):
        return cond, cond
    uncond = model.apply_model(x, sigma, ctx_neg)
    return uncond + (cond - uncond) * cfg, uncond


# ---------------------------------------------------------------------------
# Scheduler helpers (ComfyUI ``comfy/samplers.py`` schedule set)
# ---------------------------------------------------------------------------


def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def ddim_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    x = 1
    if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]

    ss = max(len(s.sigmas) // steps, 1)
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)


def normal_scheduler(model_sampling, steps, sgm=False):
    s = model_sampling
    start = s.timestep(torch.tensor([float(s.sigma_max)], dtype=torch.float64))[0]
    end = s.timestep(torch.tensor([float(s.sigma_min)], dtype=torch.float64))[0]

    append_zero_flag = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps)

    sigs = [float(s.sigma(ts)) for ts in timesteps]
    if append_zero_flag:
        sigs += [0.0]
    return torch.FloatTensor(sigs)


def sgm_uniform_scheduler(model_sampling, steps):
    return normal_scheduler(model_sampling, steps, sgm=True)


def karras_scheduler(model_sampling, steps, rho=7.0):
    """Karras et al. (2022) schedule (``get_sigmas_karras``)."""
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


def exponential_scheduler(ms, steps):
    """Log-linear ("exponential") schedule (``get_sigmas_exponential``)."""
    sigma_min = float(ms.sigma_min)
    sigma_max = float(ms.sigma_max)
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps).exp()
    return append_zero(sigmas)


def polyexponential_scheduler(ms, steps, rho=1.0):
    """Polynomial-in-log-sigma schedule (``get_sigmas_polyexponential``)."""
    sigma_min = float(ms.sigma_min)
    sigma_max = float(ms.sigma_max)
    ramp = torch.linspace(1, 0, steps) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def laplace_scheduler(ms, steps, mu=0.0, beta=0.5):
    """Laplace schedule (Tiankai et al., 2024; ``get_sigmas_laplace``)."""
    sigma_min = float(ms.sigma_min)
    sigma_max = float(ms.sigma_max)
    epsilon = 1e-5
    x = torch.linspace(0, 1, steps)
    lmb = mu - beta * torch.sign(0.5 - x) * torch.log(1 - 2 * torch.abs(0.5 - x) + epsilon)
    sigmas = torch.clamp(torch.exp(lmb), min=sigma_min, max=sigma_max)
    return append_zero(sigmas)


def beta_scheduler(model_sampling, steps, alpha=BETA_ALPHA_DEFAULT, beta=BETA_BETA_DEFAULT):
    """Beta-distributed schedule (ComfyUI ``beta_scheduler``).

    ``alpha`` and ``beta`` are the two shape parameters of the Beta distribution
    used to place the steps; ``0.6``/``0.6`` matches ComfyUI.
    """
    if _betaincinv is None:  # pragma: no cover
        raise RuntimeError("scipy is required for the beta scheduler")
    total_timesteps = len(model_sampling.sigmas) - 1
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)
    ts = np.rint(_betaincinv(alpha, beta, ts) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(model_sampling.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def linear_quadratic_scheduler(model_sampling, steps, threshold_noise=0.025, linear_steps=None):
    """Linear+quadratic schedule (ComfyUI ``linear_quadratic_schedule``)."""
    if steps == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const
            for i in range(linear_steps, steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule) * model_sampling.sigma_max.cpu()


def kl_optimal_scheduler(model_sampling, steps):
    """KL-optimal schedule (A1111 PR #15608)."""
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    adj_idxs = torch.arange(steps, dtype=torch.float).div_(steps - 1)
    sigmas = adj_idxs.new_zeros(steps + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas


# Every scheduler here is normalised to ``(model_sampling, steps)``; the dict is
# kept name -> callable so ``paths.SCHEDULERS`` and the UI stay simple.
SCHEDULER_HANDLERS = {
    "simple": simple_scheduler,
    "normal": normal_scheduler,
    "karras": karras_scheduler,
    "exponential": exponential_scheduler,
    "ddim_uniform": ddim_scheduler,
    "beta": beta_scheduler,
    "sgm_uniform": sgm_uniform_scheduler,
    "linear_quadratic": linear_quadratic_scheduler,
    "kl_optimal": kl_optimal_scheduler,
    "polyexponential": polyexponential_scheduler,
    "laplace": laplace_scheduler,
}

SCHEDULER_NAMES = list(SCHEDULER_HANDLERS)


def compute_sigmas(model_sampling, steps, name, **kwargs):
    """Sigmas for ``steps`` at ``name`` (no denoise slicing).

    ``kwargs`` are forwarded to the scheduler, so the ``beta`` scheduler can
    receive its two shape parameters (``alpha``/``beta``); every other schedule
    ignores extra keywords.
    """
    fn = SCHEDULER_HANDLERS.get(name)
    if fn is None:
        return normal_scheduler(model_sampling, steps)
    return fn(model_sampling, steps, **kwargs)


# ---------------------------------------------------------------------------
# Multistep / integral helpers
# ---------------------------------------------------------------------------


def linear_multistep_coeff(order, t, i, j):
    """k-diffusion ``linear_multistep_coeff`` via exact polynomial integration.

    The integrand is a degree ``order - 1`` polynomial in ``tau``; its product
    form is expanded and integrated analytically, so no quadrature dependency is
    needed (it matches ``scipy.integrate.quad`` for these polynomials).
    """
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")
    denom = 1.0
    for k in range(order):
        if k != j:
            denom *= (t[i - j] - t[i - k])
    poly = np.array([1.0])
    for k in range(order):
        if k == j:
            continue
        poly = np.polynomial.polynomial.polymul(poly, np.array([-t[i - k], 1.0]))
    integ = np.polynomial.polynomial.polyint(poly)
    a, b = t[i], t[i + 1]
    val = (
        np.polynomial.polynomial.polyval(b, integ)
        - np.polynomial.polynomial.polyval(a, integ)
    )
    return float(val / denom)


def _edm2t(edm_steps, epsilon_s=1e-3, sigma_min=0.002, sigma_max=80):
    """DEIS ``edm2t`` transfer between EDM sigma and DEIS ``t``."""
    edm_steps = np.asarray(edm_steps, dtype=np.float64)
    beta_d = 2 * (
        np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)
    ) / (epsilon_s - 1)
    beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * beta_d
    t_steps = (np.sqrt(beta_min ** 2 + 2 * beta_d * np.log(edm_steps ** 2 + 1)) - beta_min) / beta_d
    return t_steps, beta_min, beta_d + beta_min


def _deis_coeff_list(t_steps, max_order, num_points=10000, deis_mode="tab"):
    """DEIS coefficients (ported from ``comfy/k_diffusion/deis.py``)."""
    t_steps = np.asarray([float(x) for x in t_steps], dtype=np.float64)

    def cal_poly(prev_t, j, taus):
        poly = np.ones_like(taus)
        for k in range(prev_t.shape[0]):
            if k == j:
                continue
            poly *= (taus - prev_t[k]) / (prev_t[j] - prev_t[k])
        return poly

    if deis_mode == "tab":
        t_steps_t, beta_0, beta_1 = _edm2t(t_steps)
        coeffs = []
        for i in range(t_steps_t.shape[0] - 1):
            order = min(i + 1, max_order)
            if order == 1:
                coeffs.append([])
                continue
            t_cur, t_next = t_steps_t[i], t_steps_t[i + 1]
            taus = np.linspace(t_cur, t_next, num_points)
            dtau = (t_next - t_cur) / num_points
            prev_t = t_steps_t[[i - k for k in range(order)]]
            alpha = np.exp(-0.5 * taus ** 2 * (beta_1 - beta_0) - taus * beta_0)
            d_log_alpha_dtau = -taus * (beta_1 - beta_0) - beta_0
            integrand = -0.5 * d_log_alpha_dtau / np.sqrt(alpha * (1 - alpha))
            coeff_temp = [
                float(np.sum(integrand * cal_poly(prev_t, j, taus)) * dtau)
                for j in range(order)
            ]
            coeffs.append(coeff_temp)
        return coeffs

    # "rhoab": analytical second/third order coefficients.
    def integral_2(a, b, start, end, c):
        coeff = (end ** 3 - start ** 3) / 3 - (end ** 2 - start ** 2) * (a + b) / 2 + (end - start) * a * b
        return coeff / ((c - a) * (c - b))

    def integral_3(a, b, c, start, end, d):
        coeff = (
            (end ** 4 - start ** 4) / 4
            - (end ** 3 - start ** 3) * (a + b + c) / 3
            + (end ** 2 - start ** 2) * (a * b + a * c + b * c) / 2
            - (end - start) * a * b * c
        )
        return coeff / ((d - a) * (d - b) * (d - c))

    coeffs = []
    for i in range(len(t_steps) - 1):
        order = min(i, max_order)
        if order == 0:
            coeffs.append([])
            continue
        prev_t = t_steps[[i - k for k in range(order + 1)]]
        if order == 1:
            coeff_temp = [
                ((t_steps[i + 1] - prev_t[1]) ** 2 - (t_steps[i] - prev_t[1]) ** 2) / (2 * (t_steps[i] - prev_t[1])),
                (t_steps[i + 1] - t_steps[i]) ** 2 / (2 * (prev_t[1] - t_steps[i])),
            ]
        elif order == 2:
            coeff_temp = [
                integral_2(prev_t[1], prev_t[2], t_steps[i], t_steps[i + 1], t_steps[i]),
                integral_2(t_steps[i], prev_t[2], t_steps[i], t_steps[i + 1], prev_t[1]),
                integral_2(t_steps[i], prev_t[1], t_steps[i], t_steps[i + 1], prev_t[2]),
            ]
        else:
            coeff_temp = [
                integral_3(prev_t[1], prev_t[2], prev_t[3], t_steps[i], t_steps[i + 1], t_steps[i]),
                integral_3(t_steps[i], prev_t[2], prev_t[3], t_steps[i], t_steps[i + 1], prev_t[1]),
                integral_3(t_steps[i], prev_t[1], prev_t[3], t_steps[i], t_steps[i + 1], prev_t[2]),
                integral_3(t_steps[i], prev_t[1], prev_t[2], t_steps[i], t_steps[i + 1], prev_t[3]),
            ]
        coeffs.append(coeff_temp)
    return coeffs


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def sample_euler(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                 generator=None, guider=None, ms=None, s_churn=0.0, s_tmin=0.0,
                 s_tmax=float("inf"), s_noise=1.0):
    """Euler steps (Karras et al., 2022)."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        gamma = min(s_churn / total, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            sigma_hat = sigmas[i] * (gamma + 1)
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        else:
            sigma_hat = sigmas[i]
        denoised = guide(model, x, sigma_hat * s_in, ctx_pos, ctx_neg, cfg)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(i, denoised, x, total)
        x = x + d * (sigmas[i + 1] - sigma_hat)
    return x


def sample_euler_ancestral(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                           generator=None, guider=None, ms=None, eta=1.0, s_noise=1.0):
    """Ancestral Euler. Uses the flow-correct re-noise formula for CONST models."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    flow = _is_flow(ms)
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        if flow:
            if sigmas[i + 1] == 0:
                x = denoised
            else:
                downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
                sigma_down = sigmas[i + 1] * downstep_ratio
                alpha_ip1 = 1 - sigmas[i + 1]
                alpha_down = 1 - sigma_down
                renoise_coeff = (
                    sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2
                ) ** 0.5
                sigma_down_i_ratio = sigma_down / sigmas[i]
                x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
                if eta > 0:
                    x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        else:
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if sigma_down == 0:
                x = denoised
            else:
                d = to_d(x, sigmas[i], denoised)
                x = x + d * (sigma_down - sigmas[i]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def sample_heun(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                generator=None, guider=None, ms=None, s_churn=0.0, s_tmin=0.0,
                s_tmax=float("inf"), s_noise=1.0):
    """Heun steps (Karras et al., 2022)."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        gamma = min(s_churn / total, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = guide(model, x, sigma_hat * s_in, ctx_pos, ctx_neg, cfg)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(i, denoised, x, total)
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = guide(model, x_2, sigmas[i + 1] * s_in, ctx_pos, ctx_neg, cfg)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            x = x + (d + d_2) / 2 * dt
    return x


def sample_heunpp2(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                   generator=None, guider=None, ms=None, s_churn=0.0, s_tmin=0.0,
                   s_tmax=float("inf"), s_noise=1.0):
    """Heun++2 (MIT, sd-webui-samplers-scheduler)."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    s_end = sigmas[-1]
    for i in range(total):
        _set_step(model, i, total)
        gamma = min(s_churn / total, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = guide(model, x, sigma_hat * s_in, ctx_pos, ctx_neg, cfg)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(i, denoised, x, total)
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end:
            x = x + d * dt
        elif sigmas[i + 2] == s_end:
            x_2 = x + d * dt
            denoised_2 = guide(model, x_2, sigmas[i + 1] * s_in, ctx_pos, ctx_neg, cfg)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            w = 2 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w1 = 1 - w2
            x = x + (d * w1 + d_2 * w2) * dt
        else:
            x_2 = x + d * dt
            denoised_2 = guide(model, x_2, sigmas[i + 1] * s_in, ctx_pos, ctx_neg, cfg)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]
            x_3 = x_2 + d_2 * dt_2
            denoised_3 = guide(model, x_3, sigmas[i + 2] * s_in, ctx_pos, ctx_neg, cfg)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)
            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            w1 = 1 - w2 - w3
            x = x + (w1 * d + w2 * d_2 + w3 * d_3) * dt
    return x


def sample_dpm_2(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                 generator=None, guider=None, ms=None, s_churn=0.0, s_tmin=0.0,
                 s_tmax=float("inf"), s_noise=1.0):
    """DPM-Solver-2 / Karras Algorithm 2."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        gamma = min(s_churn / total, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = guide(model, x, sigma_hat * s_in, ctx_pos, ctx_neg, cfg)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigmas[i + 1] == 0:
            x = x + d * (sigmas[i + 1] - sigma_hat)
        else:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = guide(model, x_2, sigma_mid * s_in, ctx_pos, ctx_neg, cfg)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


def sample_dpm_2_ancestral(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                           generator=None, guider=None, ms=None, eta=1.0, s_noise=1.0):
    """Ancestral DPM-Solver-2."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    flow = _is_flow(ms)
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        d = to_d(x, sigmas[i], denoised)
        if flow:
            if sigmas[i + 1] == 0:
                x = x + d * (0 - sigmas[i])
            else:
                downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
                sigma_down = sigmas[i + 1] * downstep_ratio
                alpha_ip1 = 1 - sigmas[i + 1]
                alpha_down = 1 - sigma_down
                renoise_coeff = (
                    sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2
                ) ** 0.5
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                x_2 = x + d * (sigma_mid - sigmas[i])
                denoised_2 = guide(model, x_2, sigma_mid * s_in, ctx_pos, ctx_neg, cfg)
                d_2 = to_d(x_2, sigma_mid, denoised_2)
                x = x + d_2 * (sigma_down - sigmas[i])
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        else:
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if sigma_down == 0:
                x = x + d * (sigma_down - sigmas[i])
            else:
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                x_2 = x + d * (sigma_mid - sigmas[i])
                denoised_2 = guide(model, x_2, sigma_mid * s_in, ctx_pos, ctx_neg, cfg)
                d_2 = to_d(x_2, sigma_mid, denoised_2)
                x = x + d_2 * (sigma_down - sigmas[i]) + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def sample_lms(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
               generator=None, guider=None, ms=None, order=4):
    """Linear multistep (LMS)."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


def sample_dpmpp_2s_ancestral(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                              generator=None, guider=None, ms=None, eta=1.0, s_noise=1.0):
    """DPM-Solver++(2S) ancestral."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    flow = _is_flow(ms)
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    def lambda_fn(sigma):
        return ((1 - sigma) / sigma).log()

    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        if flow:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (
                sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2
            ) ** 0.5
            if sigmas[i + 1] == 0:
                x = denoised
            else:
                if float(sigmas[i]) == 1.0:
                    sigma_s = 0.9999
                else:
                    t_i, t_down = lambda_fn(sigmas[i]), lambda_fn(sigma_down)
                    h = t_down - t_i
                    sigma_s = sigma_fn(t_i + 0.5 * h)
                sigma_s_i_ratio = sigma_s / sigmas[i]
                u = sigma_s_i_ratio * x + (1 - sigma_s_i_ratio) * denoised
                d_i = guide(model, u, sigma_s * s_in, ctx_pos, ctx_neg, cfg)
                sigma_down_i_ratio = sigma_down / sigmas[i]
                x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * d_i
            if sigmas[i + 1] > 0 and eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
        else:
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if sigma_down == 0:
                x = x + to_d(x, sigmas[i], denoised) * (sigma_down - sigmas[i])
            else:
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                h = t_next - t
                s = t + 0.5 * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * 0.5).expm1() * denoised
                denoised_2 = guide(model, x_2, sigma_fn(s) * s_in, ctx_pos, ctx_neg, cfg)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def sample_dpmpp_sde(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                     generator=None, guider=None, ms=None, eta=1.0, s_noise=1.0, r=0.5):
    """DPM-Solver++ (stochastic)."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = guide(model, x_2, sigma_fn(s) * s_in, ctx_pos, ctx_neg, cfg)
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


def sample_dpmpp_2m(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                    generator=None, guider=None, ms=None):
    """DPM-Solver++(2M)."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


def sample_dpmpp_2m_sde(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                        generator=None, guider=None, ms=None, eta=1.0, s_noise=1.0,
                        solver_type="midpoint"):
    """DPM-Solver++(2M) SDE."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    old_denoised = None
    h_last = None
    h = None
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h
            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised
            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                else:
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
    return x


def sample_dpmpp_3m_sde(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                        generator=None, guider=None, ms=None, eta=1.0, s_noise=1.0):
    """DPM-Solver++(3M) SDE."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised
            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d
            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


def _ddpm_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = alpha_cumprod / alpha_cumprod_prev
    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu = mu + ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu


def sample_ddpm(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                generator=None, guider=None, ms=None):
    """DDPM ancestral sampler (VP/epsilon schedule only; see module note)."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        x = _ddpm_step(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x = x * torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


def sample_ipndm(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                 generator=None, guider=None, ms=None, max_order=4):
    """iPNDM (diff-sampler)."""
    guide = guider or _plain_guider
    s_in = noise.new_ones([noise.shape[0]])
    total = len(sigmas) - 1
    x_next = noise
    buffer_model = []
    for i in range(total):
        _set_step(model, i, total)
        t_cur, t_next = sigmas[i], sigmas[i + 1]
        x_cur = x_next
        denoised = guide(model, x_cur, t_cur * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x_cur, total)
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next == 0:
            x_next = denoised
        elif order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)
    return x_next


def sample_ipndm_v(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                   generator=None, guider=None, ms=None, max_order=4):
    """iPNDM with variable step sizes (diff-sampler)."""
    guide = guider or _plain_guider
    s_in = noise.new_ones([noise.shape[0]])
    total = len(sigmas) - 1
    x_next = noise
    t_steps = sigmas
    buffer_model = []
    for i in range(total):
        _set_step(model, i, total)
        t_cur, t_next = sigmas[i], sigmas[i + 1]
        x_cur = x_next
        denoised = guide(model, x_cur, t_cur * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x_cur, total)
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next == 0:
            x_next = denoised
        elif order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            h_n = t_next - t_cur
            h_n_1 = t_cur - t_steps[i - 1]
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:
            h_n = t_next - t_cur
            h_n_1 = t_cur - t_steps[i - 1]
            h_n_2 = t_steps[i - 1] - t_steps[i - 2]
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:
            h_n = t_next - t_cur
            h_n_1 = t_cur - t_steps[i - 1]
            h_n_2 = t_steps[i - 1] - t_steps[i - 2]
            h_n_3 = t_steps[i - 2] - t_steps[i - 3]
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
    return x_next


def sample_deis(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                generator=None, guider=None, ms=None, max_order=3, deis_mode="tab"):
    """DEIS (qsh-zh/deis)."""
    guide = guider or _plain_guider
    s_in = noise.new_ones([noise.shape[0]])
    total = len(sigmas) - 1
    x_next = noise
    coeff_list = _deis_coeff_list(sigmas, max_order, deis_mode=deis_mode)
    buffer_model = []
    for i in range(total):
        _set_step(model, i, total)
        t_cur, t_next = sigmas[i], sigmas[i + 1]
        x_cur = x_next
        denoised = guide(model, x_cur, t_cur * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x_cur, total)
        d_cur = (x_cur - denoised) / t_cur
        order = min(max_order, i + 1)
        if t_next <= 0:
            order = 1
        if order == 1:
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k + 1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
    return x_next


def _res_multistep(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback,
                   generator, guider, ms, s_noise, eta, cfg_pp):
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t
    old_sigma_down = None
    old_denoised = None
    for i in range(total):
        _set_step(model, i, total)
        if cfg_pp:
            denoised, uncond = _guided_cfg(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        else:
            denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
            uncond = None
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigma_down == 0 or old_denoised is None:
            if cfg_pp:
                d = to_d(x, sigmas[i], uncond)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                x = x + d * (sigma_down - sigmas[i])
        else:
            t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            if cfg_pp:
                x = x + (denoised - uncond)
                x = sigma_fn(h) * x + h * (b1 * uncond + b2 * old_denoised)
            else:
                x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        if sigma_up > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        old_denoised = uncond if cfg_pp else denoised
        old_sigma_down = sigma_down
    return x


def sample_res_multistep_ancestral(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                   callback=None, generator=None, guider=None, ms=None,
                                   eta=1.0, s_noise=1.0):
    return _res_multistep(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback,
                          generator, guider, ms, s_noise, eta, False)


def sample_res_multistep_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                callback=None, generator=None, guider=None, ms=None,
                                s_noise=1.0):
    return _res_multistep(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback,
                          generator, guider, ms, s_noise, 0.0, True)


def sample_res_multistep_ancestral_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                          callback=None, generator=None, guider=None, ms=None,
                                          eta=1.0, s_noise=1.0):
    return _res_multistep(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback,
                          generator, guider, ms, s_noise, eta, True)


def sample_gradient_estimation(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                               generator=None, guider=None, ms=None, ge_gamma=2.0, cfg_pp=False):
    """Gradient-estimation sampler (Ikeda et al.)."""
    guide = guider or _plain_guider
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    old_d = None
    for i in range(total):
        _set_step(model, i, total)
        if cfg_pp:
            denoised, uncond = _guided_cfg(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
            d = to_d(x, sigmas[i], uncond)
        else:
            denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
            d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback(i, denoised, x, total)
        dt = sigmas[i + 1] - sigmas[i]
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            if cfg_pp:
                x = denoised + d * sigmas[i + 1]
            else:
                x = x + d * dt
            if i >= 1:
                d_bar = (ge_gamma - 1) * (d - old_d)
                x = x + d_bar * dt
        old_d = d
    return x


def sample_gradient_estimation_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                      callback=None, generator=None, guider=None, ms=None,
                                      ge_gamma=2.0):
    return sample_gradient_estimation(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                      callback=callback, generator=generator, guider=guider,
                                      ms=ms, ge_gamma=ge_gamma, cfg_pp=True)


def sample_er_sde(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                  generator=None, guider=None, ms=None, s_noise=1.0, max_stage=3):
    """Extended Reverse-Time SDE (VP ER-SDE-Solver-3, arXiv 2309.06169)."""
    guide = guider or _plain_guider
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1

    def noise_scaler(v):
        return v * ((v ** 0.3).exp() + 10.0)

    num_points = 200.0
    point_indice = torch.arange(0, num_points, dtype=torch.float32, device=x.device)

    sigmas = _offset_sigma_for_snr(sigmas, ms)
    half_log_snrs = _lambda(ms, sigmas)
    er_lambdas = half_log_snrs.neg().exp()

    old_denoised = None
    old_denoised_d = None
    for i in range(total):
        _set_step(model, i, total)
        denoised = guide(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised
            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d
                if stage_used >= 3:
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d
            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x


# --- CFG++ variants -------------------------------------------------------


def sample_euler_ancestral_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                  callback=None, generator=None, guider=None, ms=None,
                                  eta=1.0, s_noise=1.0):
    """Ancestral Euler (CFG++); reads cond/uncond separately."""
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        denoised, uncond = _guided_cfg(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            alpha_s = sigmas[i] * _lambda(ms, sigmas[i]).exp()
            alpha_t = sigmas[i + 1] * _lambda(ms, sigmas[i + 1]).exp()
            d = to_d(x, sigmas[i], alpha_s * uncond)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i] / alpha_s, sigmas[i + 1] / alpha_t, eta=eta)
            sigma_down = alpha_t * sigma_down
            x = alpha_t * denoised + sigma_down * d
            if eta > 0 and s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def sample_euler_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                        generator=None, guider=None, ms=None):
    return sample_euler_ancestral_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                         callback=callback, generator=generator, guider=guider,
                                         ms=ms, eta=0.0, s_noise=0.0)


def sample_dpmpp_2s_ancestral_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg,
                                     callback=None, generator=None, guider=None, ms=None,
                                     eta=1.0, s_noise=1.0):
    noise_sampler = make_noise_sampler(noise, generator)
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    for i in range(total):
        _set_step(model, i, total)
        denoised, uncond = _guided_cfg(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback(i, denoised, x, total)
        if sigma_down == 0:
            d = to_d(x, sigmas[i], uncond)
            x = denoised + d * sigma_down
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            h = t_next - t
            s = t + 0.5 * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - uncond)) - (-h * 0.5).expm1() * denoised
            denoised_2 = model.apply_model(x_2, sigma_fn(s) * s_in, ctx_pos)
            if not math.isclose(float(cfg), 1.0):
                uncond_2 = model.apply_model(x_2, sigma_fn(s) * s_in, ctx_neg)
                denoised_2 = uncond_2 + (denoised_2 - uncond_2) * cfg
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - uncond)) - (-h).expm1() * denoised_2
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def sample_dpmpp_2m_cfg_pp(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                           generator=None, guider=None, ms=None):
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    t_fn = lambda sigma: sigma.log().neg()
    old_uncond_denoised = None
    for i in range(total):
        _set_step(model, i, total)
        denoised, uncond = _guided_cfg(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond
    return x


SAMPLERS = {
    "euler_ancestral": sample_euler_ancestral,
    "euler_ancestral_cfg_pp": sample_euler_ancestral_cfg_pp,
    "euler_cfg_pp": sample_euler_cfg_pp,
    "heun": sample_heun,
    "heunpp2": sample_heunpp2,
    "dpm_2": sample_dpm_2,
    "dpm_2_ancestral": sample_dpm_2_ancestral,
    "lms": sample_lms,
    "dpmpp_2s_ancestral": sample_dpmpp_2s_ancestral,
    "dpmpp_2s_ancestral_cfg_pp": sample_dpmpp_2s_ancestral_cfg_pp,
    "dpmpp_sde": sample_dpmpp_sde,
    "dpmpp_2m": sample_dpmpp_2m,
    "dpmpp_2m_cfg_pp": sample_dpmpp_2m_cfg_pp,
    "dpmpp_2m_sde": sample_dpmpp_2m_sde,
    "dpmpp_2m_sde_heun": functools.partial(sample_dpmpp_2m_sde, solver_type="heun"),
    "dpmpp_3m_sde": sample_dpmpp_3m_sde,
    # ComfyUI's ``*_gpu`` SDE variants differ only in the correlated-noise source
    # (a GPU Brownian tree); this module already draws noise on the latent's
    # device, so they alias the base implementations.
    "dpmpp_sde_gpu": sample_dpmpp_sde,
    "dpmpp_2m_sde_gpu": sample_dpmpp_2m_sde,
    "dpmpp_3m_sde_gpu": sample_dpmpp_3m_sde,
    "ddpm": sample_ddpm,
    "ddim": sample_euler,
    "ipndm": sample_ipndm,
    "ipndm_v": sample_ipndm_v,
    "deis": sample_deis,
    "res_multistep_ancestral": sample_res_multistep_ancestral,
    "res_multistep_cfg_pp": sample_res_multistep_cfg_pp,
    "res_multistep_ancestral_cfg_pp": sample_res_multistep_ancestral_cfg_pp,
    "gradient_estimation": sample_gradient_estimation,
    "gradient_estimation_cfg_pp": sample_gradient_estimation_cfg_pp,
    "er_sde": sample_er_sde,
}

SAMPLER_NAMES = list(SAMPLERS)