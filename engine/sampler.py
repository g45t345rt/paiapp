"""Flow-matching sampler for Anima.

Anima uses ComfyUI's ``ModelSamplingDiscreteFlow`` ("AuraFlow") sampling with
``shift=3`` and ``multiplier=1000``: the DiT is fed ``sigma * 1000`` as its
timestep, predicts velocity, and ``x0 = x - v * sigma``. The ``simple`` sigma
schedule and a plain Euler ODE step complete the recipe.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .anima_ref import DEFAULT_SLOTS, AnimaRefConditioning, activate, restore
from . import diffusion_samplers as _extra

SHIFT = 3.0
MULTIPLIER = 1.0


def reshape_sigma(sigma, ndim):
    if sigma.nelement() == 1:
        return sigma.view(())
    return sigma.view(sigma.shape[:1] + (1,) * (ndim - 1))


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class ModelSamplingFlow:
    """ComfyUI's ``ModelSamplingDiscreteFlow`` (CONST prediction)."""

    def __init__(self, shift=SHIFT, multiplier=MULTIPLIER):
        self.shift = shift
        self.multiplier = multiplier
        ts = torch.arange(1, 1001, dtype=torch.float32)
        self.sigmas = time_snr_shift(shift, ts / 1000.0)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def calculate_input(self, sigma, x):
        return x

    def calculate_denoised(self, sigma, model_output, x):
        s = reshape_sigma(sigma, model_output.ndim)
        return x - model_output * s

    def noise_scaling(self, sigma, noise, latent_image):
        s = reshape_sigma(sigma, noise.ndim)
        return s * noise + (1.0 - s) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        s = reshape_sigma(sigma, latent.ndim)
        return latent / (1.0 - s)


def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def normal_scheduler(model_sampling, steps):
    """ComfyUI ``normal`` schedule: uniform steps in timestep space."""
    s = model_sampling
    start = s.timestep(torch.tensor([float(s.sigma_max)]))[0]
    end = s.timestep(torch.tensor([float(s.sigma_min)]))[0]
    timesteps = torch.linspace(float(start), float(end), steps)
    sigs = [float(s.sigma(ts)) for ts in timesteps]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def sgm_uniform_scheduler(model_sampling, steps):
    """ComfyUI ``sgm_uniform`` schedule: uniform in timestep space, last dropped.

    The LCM / Lightning recipe uses this schedule; it matches
    :func:`normal_scheduler` but samples ``steps + 1`` points and discards the
    final one (ComfyUI's ``normal_scheduler(sgm=True)``).
    """
    s = model_sampling
    start = s.timestep(torch.tensor([float(s.sigma_max)]))[0]
    end = s.timestep(torch.tensor([float(s.sigma_min)]))[0]
    timesteps = torch.linspace(float(start), float(end), steps + 1)[:-1]
    sigs = [float(s.sigma(ts)) for ts in timesteps]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def karras_scheduler(model_sampling, steps, rho=7.0):
    """ComfyUI ``karras`` schedule (Karras et al. 2022)."""
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    ramp = torch.linspace(0, 1, steps)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sigs = ((max_inv + ramp * (min_inv - max_inv)) ** rho).tolist()
    sigs += [0.0]
    return torch.FloatTensor(sigs)


# The selectable schedules live in engine.diffusion_samplers (simple, normal,
# karras, exponential, ddim_uniform, beta, sgm_uniform, linear_quadratic,
# kl_optimal, polyexponential, laplace).
SCHEDULERS = dict(_extra.SCHEDULER_HANDLERS)


def _scheduler_kwargs(name, params):
    """Extra keyword arguments for a scheduler from the run config.

    Only the schedules with shape parameters read anything here; the rest stay
    on their defaults (which is what ComfyUI does too).
    """
    def num(key, default):
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    if name == "beta":
        return {
            "alpha": num("beta_alpha", _extra.BETA_ALPHA_DEFAULT),
            "beta": num("beta_beta", _extra.BETA_BETA_DEFAULT),
        }
    if name == "karras":
        return {"rho": num("karras_rho", 7.0)}
    if name == "polyexponential":
        return {"rho": num("poly_rho", 1.0)}
    if name == "laplace":
        return {"mu": num("laplace_mu", 0.0), "beta": num("laplace_beta", 0.5)}
    if name == "linear_quadratic":
        linear_steps = num("lq_linear_steps", 0.0)
        return {
            "threshold_noise": num("lq_threshold", 0.025),
            "linear_steps": int(linear_steps) if linear_steps > 0 else None,
        }
    return {}


def scheduler_sigmas(model_sampling, steps, denoise, name="normal", params=None):
    """Sigmas for ``steps`` at ``denoise`` for a named scheduler.

    ``params`` (usually the run config) supplies optional schedule parameters,
    e.g. the ``beta`` shape parameters or the ``karras`` rho.
    """
    if name not in _extra.SCHEDULER_HANDLERS:
        name = "normal"
    kwargs = _scheduler_kwargs(name, params or {})
    if denoise is None or denoise > 0.9999:
        return _extra.compute_sigmas(model_sampling, steps, name, **kwargs)
    if denoise <= 0.0:
        return torch.FloatTensor([])
    new_steps = int(steps / denoise)
    sigmas = _extra.compute_sigmas(model_sampling, new_steps, name, **kwargs)
    return sigmas[-(steps + 1):]


def ksampler_sigmas(model_sampling, steps, denoise):
    """Sigmas for ``steps`` at ``denoise`` (ComfyUI ``KSampler.set_steps``)."""
    return scheduler_sigmas(model_sampling, steps, denoise, "simple")


class AnimaApplyModel:
    """Wraps the DiT so ``apply_model(x, sigma, context)`` returns ``x0``.

    ``lllite`` is a list of ``{"adapter", "start", "end"}`` entries: the Anima
    ControlNet-LLLite adapters patched into the DiT together with the step range
    each applies over. :meth:`set_step` toggles them, so a run can combine, for
    example, a 3D-pose adapter (end 1.0) and a regional adapter (end 0.3).

    ``cross_attn_mask`` routes each image token to the prompt block(s) it may
    attend to, for regional prompting (see :mod:`engine.regions`).
    """

    def __init__(self, dit, dtype=torch.bfloat16, lllite=None, cross_attn_mask=None):
        self.dit = dit
        self.dtype = dtype
        self.lllite = lllite or []
        self.cross_attn_mask = cross_attn_mask

    def set_step(self, index, total):
        frac = index / max(1, total)
        for entry in self.lllite:
            adapter = entry.get("adapter")
            if adapter is None:
                continue
            adapter.set_active(entry.get("start", 0.0) <= frac <= entry.get("end", 1.0))

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = (sigma * MULTIPLIER).float()
        out = self.dit(
            x.to(self.dtype), t, context.to(self.dtype),
            cross_attn_mask=self.cross_attn_mask,
        )
        return x - out.float() * s


def _guider(model, x, sigma, ctx_pos, ctx_neg, cfg):
    cond = model.apply_model(x, sigma, ctx_pos)
    if math.isclose(cfg, 1.0):
        return cond
    uncond = model.apply_model(x, sigma, ctx_neg)
    return uncond + (cond - uncond) * cfg


class AnimaRefApplyModel(AnimaApplyModel):
    """Anima apply-model that feeds a reference frame when the context carries one.

    The context is an :class:`engine.anima_ref.AnimaRefConditioning`; a forward
    with a non-empty ``context.refs`` attaches the reference frame(s) (and
    activates the adapter LoRA for that forward), while empty ``context.refs``
    runs the plain single-frame path with the LoRA off. The step scheduling of
    :class:`AnimaApplyModel` (ControlNet-LLLite) is inherited unchanged.
    """

    def __init__(self, dit, dtype=torch.bfloat16, lllite=None, cross_attn_mask=None,
                 adapter=None, offset=1.0):
        super().__init__(dit, dtype, lllite, cross_attn_mask)
        self.adapter = adapter
        self.offset = float(offset)

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = (sigma * MULTIPLIER).float()
        if isinstance(context, AnimaRefConditioning):
            text, refs = context.text, context.refs
        else:
            text, refs = context, {}
        if refs and self.adapter is not None and getattr(self.adapter, "asym", False):
            # v2: the reference stream is cached (keys/values per block) and the
            # target attends to it, so only the target frame is forwarded. The
            # asymmetric adapter reads one (character) reference.
            ref = refs.get("character")
            if ref is None:
                ref = next(iter(refs.values()))
            ref_kv = self.adapter.ref_kv(self.dit, ref.to(self.dtype), x, self.offset)
            previous = activate(self.adapter)
            try:
                out = self.dit.forward_asym(
                    x.to(self.dtype), t, text.to(self.dtype), ref_kv
                )
            finally:
                restore(previous)
        elif refs:
            # Pilot (and any multi-slot adapter): one clean reference frame per
            # filled slot, at the slot's RoPE t index and offset.
            slots = self.adapter.slots if self.adapter is not None else DEFAULT_SLOTS
            hp = -(-x.shape[-2] // self.dit.patch_spatial)
            frames = []
            for slot, spec in slots.items():
                latent = refs.get(slot)
                if latent is None:
                    continue
                frames.append((
                    latent.to(self.dtype),
                    int(spec["t"]),
                    round(float(spec["offset_h"]) * self.offset * hp),
                    int(spec.get("offset_w", 0)),
                ))
            if frames:
                previous = activate(self.adapter)
                try:
                    out = self.dit.forward_with_refs(
                        x.to(self.dtype), t, text.to(self.dtype), frames,
                        cross_attn_mask=self.cross_attn_mask,
                    )
                finally:
                    restore(previous)
            else:
                out = self.dit(
                    x.to(self.dtype), t, text.to(self.dtype),
                    cross_attn_mask=self.cross_attn_mask,
                )
        else:
            out = self.dit(
                x.to(self.dtype), t, text.to(self.dtype),
                cross_attn_mask=self.cross_attn_mask,
            )
        return x - out.float() * s


def anima_ref_guider(model, x, sigma, ctx_pos, ctx_neg, cfg, ref_guidance=4.5,
                     correction="apg", rescale_phi=0.5):
    """Separate text/reference guidance with APG or CFG-rescale correction.

    Ports the ComfyUI "Anima Ref Guider": the ``negative_base`` branch has no
    reference frame, so ``guided = base + ref_guidance * (neg - base) + cfg *
    (pos - neg)`` steers the reference and text effects independently. At
    ``ref_guidance == 1`` the reference is in both CFG branches and the base
    branch is skipped (plain CFG). ``correction`` is ``apg``, ``rescale`` or
    ``none``.
    """
    pos = model.apply_model(x, sigma, ctx_pos)
    neg = model.apply_model(x, sigma, ctx_neg)
    if math.isclose(ref_guidance, 1.0):
        base = neg
    else:
        base_ctx = ctx_neg.without_ref() if isinstance(ctx_neg, AnimaRefConditioning) \
            else ctx_neg
        base = model.apply_model(x, sigma, base_ctx)
    guided = base + ref_guidance * (neg - base) + cfg * (pos - neg)
    if correction == "apg":
        flat = tuple(range(1, pos.ndim))
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        parallel = (update * unit).sum(dim=flat, keepdim=True) * unit
        guided = pos + (update - parallel).to(pos.dtype)
    elif correction == "rescale":
        flat = tuple(range(1, pos.ndim))
        ratio = pos.std(dim=flat, keepdim=True) / guided.std(dim=flat, keepdim=True)
        guided = rescale_phi * guided * ratio + (1 - rescale_phi) * guided
    return guided


def _set_step(model, index, total):
    """Hand the current step to a model so control adapters can schedule by it."""
    setter = getattr(model, "set_step", None)
    if setter is not None:
        setter(index, total)


def sample_euler(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                 generator=None, guider=_guider, ms=None, post_step=None):
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        sigma_hat = sigmas[i]
        denoised = guider(model, x, sigma_hat * s_in, ctx_pos, ctx_neg, cfg)
        d = (x - denoised) / reshape_sigma(sigma_hat, x.ndim)
        if callback is not None:
            callback(i, denoised, x, total)
        x = x + d * (sigmas[i + 1] - sigma_hat)
        if post_step is not None:
            x = post_step(i, x, sigmas[i + 1])
    return x



def sample_lcm(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
               generator=None, guider=_guider, ms=None, post_step=None):
    """ComfyUI / k-diffusion ``lcm`` sampler (LCM-LoRA, SDXL-Turbo-LCM).

    Each step jumps straight to the guided ``x0`` prediction and then re-noises
    with ``sigma_next``; this is what makes 4-8 step LCM sampling work. The
    re-noising noise is drawn from ``generator`` when given so a run is
    reproducible.
    """
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    for i in range(total):
        _set_step(model, i, total)
        denoised = guider(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        if callback is not None:
            callback(i, denoised, x, total)
        x = denoised
        if sigmas[i + 1] > 0:
            noise_i = torch.randn(
                x.shape, generator=generator, dtype=torch.float32, device="cpu"
            ).to(x.device, x.dtype)
            x = x + reshape_sigma(sigmas[i + 1], x.ndim) * noise_i
        if post_step is not None:
            x = post_step(i, x, sigmas[i + 1])
    return x


def sample_res_multistep(model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=None,
                         generator=None, guider=_guider, ms=None, post_step=None):
    """ComfyUI / k-diffusion ``res_multistep`` sampler (KiSS, arXiv 2308.02157).

    A second-order variable-step linear multistep method: it reuses the previous
    step's denoised prediction and its ``sigma_down`` to take a higher-order step
    (falling back to Euler on the first and last steps). ``eta=0`` means no noise
    is injected, so the result is deterministic.
    """
    x = noise
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    phi1 = lambda t: torch.expm1(t) / t
    phi2 = lambda t: (phi1(t) - 1.0) / t
    old_sigma_down = None
    old_denoised = None
    for i in range(total):
        _set_step(model, i, total)
        denoised = guider(model, x, sigmas[i] * s_in, ctx_pos, ctx_neg, cfg)
        sigma_down = sigmas[i + 1]
        if callback is not None:
            callback(i, denoised, x, total)
        if float(sigma_down) == 0.0 or old_denoised is None:
            # Euler step (first step, or the final step down to sigma 0).
            d = (x - denoised) / reshape_sigma(sigmas[i], x.ndim)
            x = x + d * (sigma_down - sigmas[i])
        else:
            # Second-order variable-step multistep update in -log(sigma) space.
            t = -torch.log(sigmas[i])
            t_old = -torch.log(old_sigma_down)
            t_next = -torch.log(sigma_down)
            t_prev = -torch.log(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1_val, phi2_val = phi1(-h), phi2(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            x = torch.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
        old_sigma_down = sigma_down
        if post_step is not None:
            x = post_step(i, x, sigma_down)
    return x


SAMPLERS = {
    "euler": sample_euler,
    "lcm": sample_lcm,
    "res_multistep": sample_res_multistep,
    **_extra.SAMPLERS,
}


def latent_mask(mask, feather, latent):
    """A selection mask as latent-grid weights (0 = keep source, 1 = regenerate).

    ``mask`` is a document-sized 8-bit image; it is downscaled to the latent
    grid, and ``feather`` (in document pixels) is blurred in *latent* space so
    the soft edge the sampler blends with survives the 8x/16x downsample instead
    of collapsing to a hard step. The result broadcasts over the batch and
    channels, so it fits both 4-D ``[1, C, H, W]`` and 5-D ``[1, C, 1, H, W]``
    latents.
    """
    height, width = int(latent.shape[-2]), int(latent.shape[-1])
    small = mask.convert("L").resize((width, height), Image.BILINEAR)
    if feather:
        scale = mask.width / max(1, width)
        small = small.filter(ImageFilter.GaussianBlur(max(0.1, float(feather) / scale)))
    weights = np.asarray(small).astype("float32") / 255.0
    shaped = weights.reshape((1, 1) + tuple(latent.shape[2:]))
    return torch.from_numpy(shaped).to(device=latent.device, dtype=latent.dtype)


class InpaintModel:
    """Wraps an apply-model so pixels outside a latent ``mask`` keep the source.

    This is the standard latent-mask inpaint: at every step the wrapped model is
    fed the source latent noised to the current sigma outside the mask, and its
    ``x0`` prediction is pinned back to the source there. A soft (feathered)
    mask blends both, so the seam is a gradient rather than a hard cut.
    """

    def __init__(self, model, ms, latent_image, noise, mask):
        self._model = model
        self._ms = ms
        self._latent = latent_image
        self._noise = noise
        self._mask = mask

    def set_step(self, index, total):
        """Forward the step to the wrapped model, for control scheduling."""
        setter = getattr(self._model, "set_step", None)
        if setter is not None:
            setter(index, total)

    def apply_model(self, x, sigma, ctx):
        keep = 1.0 - self._mask
        clean = self._ms.noise_scaling(sigma, self._noise, self._latent).to(x.dtype)
        out = self._model.apply_model(x * self._mask + clean * keep, sigma, ctx)
        return out * self._mask + self._latent.to(out.dtype) * keep


class _Keep(Exception):
    """Internal: raised from a wrapped step callback to stop early and keep x0."""


def sample(ms, model, noise, latent_image, sigmas, ctx_pos, ctx_neg, cfg,
           sampler="euler", callback=None, generator=None, mask=None,
           feather=0.0, guider=_guider, cancel=None, post_step=None):
    """KSampler.sample flow, returning the finished latent.

    With ``mask`` (a document-sized selection image) the run is a masked inpaint
    rather than a plain img2img: only the masked region is regenerated each step
    and the rest is held at ``latent_image`` (see :class:`InpaintModel`).

    ``guider`` is the per-step guidance function (default: plain CFG). The Anima
    reference adapter passes :func:`anima_ref_guider` to add its separate
    reference guidance and correction.

    ``cancel`` enables the "stop and keep" path: when its ``keep_requested`` flag
    is set, the sampler stops after the current step and returns that step's
    ``x0`` estimate rather than finishing, so the partial image is decoded and
    delivered like any other result. This works for every sampler that calls
    ``callback(i, denoised, x, total)``; a hard :meth:`CancelToken.cancel` still
    raises :class:`Cancelled` from the callback and is never swallowed here.
    """
    step = SAMPLERS.get(sampler, sample_euler)
    if mask is not None:
        model = InpaintModel(
            model, ms, latent_image, noise, latent_mask(mask, feather, latent_image)
        )
    keep = {"denoised": None}

    def _wrapped(i, denoised, *rest):
        if denoised is not None:
            keep["denoised"] = denoised
        if callback is not None:
            callback(i, denoised, *rest)
        if cancel is not None and cancel.keep_requested and keep["denoised"] is not None:
            raise _Keep()

    wrapped = _wrapped if (callback is not None or cancel is not None) else None
    noise = ms.noise_scaling(sigmas[0], noise, latent_image)
    # ``post_step`` is only meaningful to the built-in samplers; the plug-in
    # schedules in engine.diffusion_samplers do not take it, so pass it only when
    # a caller actually supplied one.
    extra = {"post_step": post_step} if post_step is not None else {}
    try:
        samples = step(
            model, noise, sigmas, ctx_pos, ctx_neg, cfg, callback=wrapped,
            generator=generator, guider=guider, ms=ms, **extra,
        )
    except _Keep:
        samples = keep["denoised"]
    return ms.inverse_noise_scaling(sigmas[-1], samples)


class ModelSamplingDiscrete:
    """ComfyUI's ``ModelSamplingDiscrete`` (epsilon prediction, SD 1.x/SDXL)."""

    def __init__(self, linear_start=0.00085, linear_end=0.012, timesteps=1000):
        betas = torch.linspace(
            linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64
        ) ** 2
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sigmas = (((1 - alphas_cumprod) / alphas_cumprod) ** 0.5).float()
        self.log_sigmas = self.sigmas.log()

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        sigma = torch.as_tensor(sigma).reshape(-1).float()
        dists = sigma[:, None].log() - self.log_sigmas[None, :].to(sigma.device)
        return dists.abs().argmin(dim=1).view(sigma.shape)

    def sigma(self, timestep):
        t = torch.clamp(torch.as_tensor(timestep).float(), 0, len(self.sigmas) - 1)
        low = t.floor().long()
        high = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low] + w * self.log_sigmas[high]
        return log_sigma.exp()

    def calculate_input(self, sigma, x):
        s = reshape_sigma(sigma, x.ndim)
        return x / (s ** 2 + 1.0).sqrt()

    def calculate_denoised(self, sigma, model_output, x):
        s = reshape_sigma(sigma, model_output.ndim)
        return x - model_output * s

    def noise_scaling(self, sigma, noise, latent_image):
        s = reshape_sigma(sigma, noise.ndim)
        return s * noise + latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class UNetApplyModel:
    """Wraps an SD/SDXL UNet so ``apply_model(x, sigma, ctx)`` returns ``x0``.

    ``controlnets`` is a list of control entries, each a dict with ``model`` (a
    diffusers ``ControlNetModel``), ``cond`` (the prepared ``[B, 3, H, W]`` hint),
    ``scale`` (its conditioning strength) and the ``start``/``end`` fractions of
    the run it applies over. Every active ControlNet is run first each step and
    its down/middle residuals are summed into the UNet, so several can steer one
    run at once. Running them per call (rather than once per step for a batched
    pass) matches diffusers, which feeds each the condition's own text embeddings
    for the conditional and unconditional passes alike.
    """

    def __init__(self, unet, ms, dtype=torch.float16, time_ids=None,
                 controlnets=None):
        self.unet = unet
        self.ms = ms
        self.dtype = dtype
        self.time_ids = time_ids
        self.controlnets = controlnets or []

    def set_step(self, index, total):
        """Mark each ControlNet active or not for the current sampling step."""
        frac = index / max(1, total)
        for entry in self.controlnets:
            entry["active"] = entry.get("start", 0.0) <= frac <= entry.get("end", 1.0)

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        model_input = self.ms.calculate_input(sigma, x)
        t = self.ms.timestep(torch.as_tensor(sigma).reshape(-1)).to(x.device)
        added = None
        pooled = context.get("pooled")
        if pooled is not None:
            added = {"text_embeds": pooled.to(self.dtype), "time_ids": self.time_ids.to(self.dtype)}
        prompt_embeds = context["prompt_embeds"].to(self.dtype)

        down_total = None
        mid_total = None
        for entry in self.controlnets:
            if not entry.get("active", True):
                continue
            controlnet = entry.get("model")
            cond = entry.get("cond")
            if controlnet is None or cond is None:
                continue
            down_blocks, mid_block = controlnet(
                model_input.to(self.dtype),
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond.to(self.dtype),
                conditioning_scale=entry.get("scale", 1.0),
                added_cond_kwargs=added,
                return_dict=False,
            )
            if down_total is None:
                down_total = [sample.to(self.dtype) for sample in down_blocks]
                mid_total = mid_block.to(self.dtype)
            else:
                down_total = [
                    total + sample.to(self.dtype)
                    for total, sample in zip(down_total, down_blocks)
                ]
                mid_total = mid_total + mid_block.to(self.dtype)

        residuals = {}
        if down_total is not None:
            residuals = {
                "down_block_additional_residuals": down_total,
                "mid_block_additional_residual": mid_total,
            }

        out = self.unet(
            model_input.to(self.dtype),
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added,
            **residuals,
        ).sample
        return self.ms.calculate_denoised(s, out.float(), x)


def flux_time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    """Qwen-Image scheduler ``mu``: linear in token count, 0.69 at 1024x1024."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class ModelSamplingFlux(ModelSamplingFlow):
    """Flux/Qwen-Image flow sampling: ``sigma = flux_time_shift(mu, 1, t)``."""

    def __init__(self, shift=1.15, timesteps=10000):
        self.shift = shift
        self.multiplier = MULTIPLIER
        self.sigmas = torch.tensor(
            [flux_time_shift(shift, 1.0, float(ts)) for ts in (torch.arange(1, timesteps + 1) / timesteps)],
            dtype=torch.float32,
        )

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)


def _split_region(context, arity):
    """Separate a family's conditioning from an optional trailing regional mask.

    A regional run wraps the conditioning as ``(..., mask)``, one element longer
    than the ``arity`` elements the family normally carries, so the mask can be
    carried with the context and differ between the conditional and unconditional
    passes (whose text block lengths differ). Returns ``(conditioning, mask)`` and
    ``mask`` is ``None`` for a non-regional run.

    The conditioning is the single tensor for an arity-1 family (Qwen-Image,
    Z-Image) and the ``arity``-tuple of tensors otherwise, so callers can unpack
    it directly.
    """
    if isinstance(context, tuple) and len(context) == arity + 1:
        cond = context[:-1]
        return (cond[0] if arity == 1 else cond), context[-1]
    return context, None


class QwenConditioning:
    """One Qwen-Image pass's conditioning: text context, optional reference
    latents and an optional regional attention mask.

    The conditional pass carries the reference latents; the unconditional pass
    omits them, matching the reference guider (its negative prompt has no
    condition images). Bundling the three keeps them together through ``sample``
    and the inpaint wrapper, which only forward ``context``.
    """

    def __init__(self, text, ref=None, mask=None):
        self.text = text
        self.ref = ref
        self.mask = mask


class QwenImageApplyModel:
    """Wraps the Qwen-Image MMDiT so ``apply_model(x, sigma, ctx)`` returns ``x0``.

    A regional run passes ``(context, mask)``; the joint attention mask routes
    each image token to its prompt block (see :mod:`engine.regions`). A
    reference-image run passes a :class:`QwenConditioning` carrying the text
    context, the condition latents and the regional mask.
    """

    def __init__(self, dit, dtype=torch.bfloat16):
        self.dit = dit
        self.dtype = dtype

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = torch.as_tensor(sigma).reshape(-1).float()
        if isinstance(context, QwenConditioning):
            text, ref, mask = context.text, context.ref, context.mask
        else:
            ref = None
            text, mask = _split_region(context, 1)
        out = self.dit(x.to(self.dtype), t, text.to(self.dtype),
                       cross_attn_mask=mask, ref=ref)
        return x - out.float() * s


class ZImageApplyModel:
    """Wraps the Z-Image NextDiT so ``apply_model(x, sigma, ctx)`` returns ``x0``.

    With ``control_latent`` set, the DiT's ControlNet branch steers every step:
    the control image's VAE latent is the hint and ``control_scale`` its strength
    (0 disables the effect).
    """

    def __init__(self, dit, dtype=torch.bfloat16, control_latent=None, control_scale=1.0,
                 control_start=0.0, control_end=1.0):
        self.dit = dit
        self.dtype = dtype
        self.control_latent = control_latent
        self.control_scale = control_scale
        self._base_scale = control_scale
        self.control_start = control_start
        self.control_end = control_end

    def set_step(self, index, total):
        """Scale the control branch to 0 outside its start/end range."""
        frac = index / max(1, total)
        self.control_scale = (
            self._base_scale
            if self.control_start <= frac <= self.control_end else 0.0
        )

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = torch.as_tensor(sigma).reshape(-1).float()
        ctx, mask = _split_region(context, 1)
        out = self.dit(
            x.to(self.dtype), t, ctx.to(self.dtype),
            control_latent=self.control_latent, control_scale=self.control_scale,
            cross_attn_mask=mask,
        )
        return x - out.float() * s


class Krea2ApplyModel:
    """Wraps the Krea 2 MMDiT so ``apply_model(x, sigma, ctx)`` returns ``x0``.

    ``context`` is a ``(text_stack, text_mask)`` pair: the flattened 12-layer
    Qwen3-VL hidden-state stack and its boolean token mask (see
    :mod:`engine.krea2_text`). A regional run appends a joint attention mask as a
    third element.
    """

    def __init__(self, dit, dtype=torch.bfloat16, ref_kv_cache=None):
        self.dit = dit
        self.dtype = dtype
        self.ref_kv_cache = ref_kv_cache

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = torch.as_tensor(sigma).reshape(-1).float()
        ctx, region_mask = _split_region(context, 2)
        hidden, mask = ctx
        out = self.dit(
            x.to(self.dtype), t, hidden.to(self.dtype),
            attention_mask=mask.to(x.device) if mask is not None else None,
            cross_attn_mask=region_mask,
            ref_kv_cache=self.ref_kv_cache,
        )
        return x - out.float() * s


class FluxApplyModel:
    """Wraps the Flux DiT so ``apply_model(x, sigma, ctx)`` returns ``x0``.

    ``context`` is a ``(t5_context, clip_pooled)`` pair; a regional run appends a
    joint attention mask as a third element. ``guidance`` is the baked-in Flux-dev
    guidance scale (a 1-element tensor) or ``None``.
    """

    def __init__(self, dit, dtype=torch.bfloat16, guidance=None):
        self.dit = dit
        self.dtype = dtype
        self.guidance = guidance

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = torch.as_tensor(sigma).reshape(-1).float()
        ctx, mask = _split_region(context, 2)
        txt, pooled = ctx
        out = self.dit(
            x.to(self.dtype), t, txt.to(self.dtype),
            y=pooled.to(self.dtype) if pooled is not None else None,
            guidance=self.guidance, cross_attn_mask=mask,
        )
        return x - out.float() * s


def _moss_context(context):
    """Unpack a MOSS ``context`` into ``(embeds, mask, skip_layers)``.

    A bare tensor (positive, no regional mask) and a ``(embeds, region_mask)``
    pair (positive regional run) skip no block; only the explicit
    ``(embeds, mask, skip_layers)`` triple carries a skip set -- that is the CFG
    negative branch, which skips the reference block 22 on the uncond pass (see
    :class:`MossImageApplyModel`). The positive pass must run every block.
    """
    if isinstance(context, tuple):
        if len(context) == 3:
            return context[0], context[1], context[2]
        if len(context) == 2:
            return context[0], context[1], ()
        return context[0], None, ()
    return context, None, ()


class MossImageApplyModel:
    """Wraps the MOSS Image DiT so ``apply_model(x, sigma, ctx)`` returns ``x0``.

    ``context`` is the Qwen3.5 hidden states (see :func:`_moss_context` for the
    regional / negative forms). Only the CFG negative branch skips a block, and
    it carries that skip set in its context triple (the reference default is
    block 22, which keeps two forwards per step while dropping one block's
    compute on the uncond pass).
    """

    def __init__(self, dit, dtype=torch.bfloat16):
        self.dit = dit
        self.dtype = dtype

    def apply_model(self, x, sigma, context):
        s = reshape_sigma(sigma, x.ndim)
        t = torch.as_tensor(sigma).reshape(-1).float()
        embeds, mask, skip = _moss_context(context)
        out = self.dit(
            x.to(self.dtype), t, embeds.to(self.dtype),
            cross_attn_mask=mask, skip_layers=skip,
        )
        return x - out.float() * s

