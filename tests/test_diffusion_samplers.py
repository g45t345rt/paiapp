"""Unit tests for the extra generation samplers and schedulers.

Run: .venv\\Scripts\\python.exe tests\\test_diffusion_samplers.py

Every scheduler in ``paths.SCHEDULERS`` must produce a valid descending sigma
schedule for both a flow and a discrete model sampling, and every sampler in
``paths.SAMPLERS`` must run end-to-end on a trivial identity apply-model and
return a finite latent of the right shape. No real DiT/VAE is needed.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import paths  # noqa: E402
from engine import sampler as S  # noqa: E402
from engine.sampler import ModelSamplingFlow, ModelSamplingDiscrete  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _IdentityModel:
    """A stand-in apply-model: its ``x0`` prediction is the input itself."""

    def apply_model(self, x, sigma, context):
        return x


def test_scheduler_lists_match():
    for name in paths.SCHEDULERS:
        check(f"scheduler registered {name}", name in S.SCHEDULERS, True)
    for name in paths.SAMPLERS:
        check(f"sampler registered {name}", name in S.SAMPLERS, True)


def test_schedules_valid():
    for ms_name, ms in (("flow", ModelSamplingFlow()),
                        ("discrete", ModelSamplingDiscrete())):
        for name in paths.SCHEDULERS:
            sigmas = S.scheduler_sigmas(ms, 6, 1.0, name)
            # ddim_uniform/beta trim duplicate steps, so the count is at least
            # one step but is not guaranteed to be exactly ``steps + 1``.
            shape_ok = len(sigmas) >= 2
            last_zero = bool(last_is_zero(sigmas))
            descending = all(
                float(sigmas[i]) >= float(sigmas[i + 1]) - 1e-6
                for i in range(len(sigmas) - 1)
            )
            finite = bool(torch.isfinite(sigmas).all())
            nonneg = bool((sigmas >= -1e-6).all())
            check(f"{ms_name} {name} len", shape_ok, True)
            check(f"{ms_name} {name} last 0", last_zero, True)
            check(f"{ms_name} {name} descending", descending, True)
            check(f"{ms_name} {name} finite", finite, True)
            check(f"{ms_name} {name} nonneg", nonneg, True)


def last_is_zero(sigmas):
    return len(sigmas) > 0 and abs(float(sigmas[-1])) < 1e-6


def test_denoise_slicing():
    ms = ModelSamplingFlow()
    full = S.scheduler_sigmas(ms, 8, 1.0, "karras")
    half = S.scheduler_sigmas(ms, 4, 0.5, "karras")
    check("denoise keeps steps+1", len(half), 5)
    check("denoise ends at 0", last_is_zero(half), True)


def test_every_sampler_runs():
    steps = 5
    for ms_name, ms in (("flow", ModelSamplingFlow()),):
        for sched in ("simple", "karras", "exponential", "beta", "kl_optimal"):
            sigmas = S.scheduler_sigmas(ms, steps, 1.0, sched).to("cpu")
            for name in paths.SAMPLERS:
                latent = torch.zeros(1, 4, 8, 8)
                gen = torch.manual_seed(1234)
                noise = torch.randn(latent.shape, generator=gen, dtype=torch.float32)
                out = S.sample(
                    ms, _IdentityModel(), noise, latent, sigmas, None, None,
                    1.0, sampler=name, generator=torch.manual_seed(1234),
                )
                ok_shape = tuple(out.shape) == tuple(latent.shape)
                ok_finite = bool(torch.isfinite(out).all())
                check(f"{name}/{sched} shape", ok_shape, True)
                check(f"{name}/{sched} finite", ok_finite, True)


def test_deterministic_when_seeded():
    ms = ModelSamplingFlow()
    sigmas = S.scheduler_sigmas(ms, 5, 1.0, "normal").to("cpu")
    latent = torch.zeros(1, 4, 8, 8)
    for name in ("euler", "dpmpp_2m", "res_multistep", "euler_ancestral"):
        def run():
            return S.sample(
                ms, _IdentityModel(),
                torch.randn(latent.shape, generator=torch.manual_seed(9), dtype=torch.float32),
                latent, sigmas, None, None, 1.0, sampler=name,
                generator=torch.manual_seed(9),
            )
        check(f"{name} seeded reproducible", bool(torch.equal(run(), run())), True)


def test_scheduler_shape_params():
    ms = ModelSamplingFlow()
    cases = [
        ("beta", {"beta_alpha": 0.5, "beta_beta": 0.7}),
        ("karras", {"karras_rho": 4.0}),
        ("polyexponential", {"poly_rho": 3.0}),
        ("laplace", {"laplace_mu": 1.0, "laplace_beta": 1.2}),
        ("linear_quadratic", {"lq_threshold": 0.05, "lq_linear_steps": 3}),
    ]
    for name, params in cases:
        base = S.scheduler_sigmas(ms, 8, 1.0, name)
        tweaked = S.scheduler_sigmas(ms, 8, 1.0, name, params)
        check(f"{name} params change schedule", bool(torch.equal(base, tweaked)), False)
        check(f"{name} params finite", bool(torch.isfinite(tweaked).all()), True)
        check(f"{name} params last 0", last_is_zero(tweaked), True)


def test_defaults_expose_shape_params():
    for key in ("beta_alpha", "beta_beta", "karras_rho", "poly_rho",
                "laplace_mu", "laplace_beta", "lq_threshold", "lq_linear_steps"):
        check(f"default has {key}", key in paths.DEFAULTS, True)
        check(f"preset field {key}", key in paths._PRESET_FIELDS, True)


def main():
    for fn in (test_scheduler_lists_match, test_schedules_valid,
               test_denoise_slicing, test_scheduler_shape_params,
               test_defaults_expose_shape_params, test_every_sampler_runs,
               test_deterministic_when_seeded):
        fn()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print(" -", line)
        sys.exit(1)
    print("\nall diffusion sampler tests passed")


if __name__ == "__main__":
    main()