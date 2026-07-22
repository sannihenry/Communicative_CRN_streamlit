#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: physics.py
#
# Physics module for:
#   "Physics-Informed Communicative Reinforcement Learning via Hamilton-Jacobi
#    Optimal Control for Robust Anatomical Landmark Detection in Low-Res Brain MRI"
#
# WHAT THIS IS (and what the two source PDFs got wrong):
# -----------------------------------------------------------------------------
# The proposal calls the constraint  ||grad V|| * F = 1  a "Hamilton-Jacobi-Bellman"
# equation and claims it is "the same principle that underlies RL". That is not
# quite right. This equation is the *stationary Hamilton-Jacobi (Eikonal)* PDE. It
# is the HJ equation for the minimum-arrival-time / geodesic problem, NOT the
# Hamilton-Jacobi-*Bellman* equation whose solution is the RL value function.
# They are cousins, not twins (see EVALUATION.md).
#
# The *correct and defensible* way to use it here is:
#   1. Precompute, ONCE per image (offline), the geodesic navigation potential V
#      by solving the Eikonal equation from each landmark, with an edge-modulated
#      speed field F. (Solving a PINN per RL step, as the PDF suggests, is both
#      unnecessary and prohibitively expensive.)
#   2. Use -V as a *potential-based reward shaping* term  (Ng, Harada & Russell,
#      1999). This provably leaves the optimal policy unchanged while giving the
#      agent a dense, anatomically-aware learning signal. The PDF's ad-hoc
#      R = -dd - lambda||grad V|| + gamma C is NOT policy-invariant and can bias
#      the optimum.
#   3. Optionally expose V (and other derived maps) as extra input channels.
#
# This module implements 1-3 with pure numpy/scipy/skimage and is fully runnable
# and cache-backed. No autograd PINN is required.
# -----------------------------------------------------------------------------

import os
import hashlib
import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.graph import MCP_Geometric
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False


# =============================================================================
# Low-level field operators
# =============================================================================
def _normalize(vol, lo_pct=1.0, hi_pct=99.0):
    vol = vol.astype(np.float32)
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def edge_strength(vol, sigma=1.0):
    """G(x) = |grad I| of a lightly smoothed, normalized volume."""
    im = ndi.gaussian_filter(_normalize(vol), sigma)
    grads = np.gradient(im)
    return np.sqrt(sum(g ** 2 for g in grads)).astype(np.float32)


def speed_field(vol, beta=8.0, sigma=1.0):
    """F(x) = 1 / (1 + beta |grad I|), in (0, 1].

    Fast (F->1) in smooth tissue, slow (F->0) at anatomical boundaries. This is
    the standard geodesic-active-contour speed and is bounded, unlike the PDF's
    alternative F = exp(-alpha G) which is also fine but less numerically tame.
    """
    g = edge_strength(vol, sigma)
    g = g / (g.max() + 1e-6)
    return (1.0 / (1.0 + beta * g)).astype(np.float32)


def hessian_vesselness(vol, sigma=1.5):
    """Cheap Frangi-like ridge/vesselness proxy from Hessian eigenvalues.

    NOTE: for brain landmark detection this channel is of dubious value (it was
    designed for tubular vessels); included for completeness / ablation only.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    n = _normalize(vol)
    H = hessian_matrix(n, sigma=sigma, use_gaussian_derivatives=False)
    eigs = hessian_matrix_eigvals(H)
    l_small = np.abs(eigs[-1])
    l_large = np.abs(eigs[0]) + 1e-6
    return np.clip(l_small / l_large, 0, 3).astype(np.float32)


# =============================================================================
# The Eikonal / stationary Hamilton-Jacobi solver
# =============================================================================
def geodesic_potential(vol, seed_xyz, beta=8.0, sigma=1.0):
    """Solve  ||grad V|| * F = 1  from `seed_xyz` (a landmark).

    Implemented as a grid minimum-cost distance with slowness cost = 1/F, which
    is exactly the first-order discretization of the Eikonal equation (this is
    what fast-marching computes). Returns the geodesic arrival-time field V.

    Falls back to a plain distance transform if scikit-image is unavailable.
    """
    seed = tuple(int(round(c)) for c in seed_xyz)
    if not _HAVE_SKIMAGE:
        m = np.ones(vol.shape, dtype=bool)
        m[seed] = False
        return ndi.distance_transform_edt(m).astype(np.float32)
    F = speed_field(vol, beta, sigma)
    cost = (1.0 / (F + 1e-6)).astype(np.float64)
    mcp = MCP_Geometric(cost)
    V, _ = mcp.find_costs([seed])
    V = np.asarray(V, dtype=np.float32)
    finite = np.isfinite(V)
    V[~finite] = V[finite].max() if finite.any() else 0.0
    return V


def euclidean_potential(shape, seed_xyz, spacing=(1, 1, 1)):
    """d(x): straight-line distance to the landmark.

    This is what the PDF's L_Consistency = ||V - d||^2 pulls V toward. Because V
    is *geodesic*, d != V wherever F varies, so that loss fights the Eikonal
    constraint. Use `geodesic` targets for consistency instead (see below).
    """
    m = np.ones(shape, dtype=bool)
    m[tuple(int(round(c)) for c in seed_xyz)] = False
    return ndi.distance_transform_edt(m, sampling=spacing).astype(np.float32)


# =============================================================================
# Per-image physics provider (cached) -- the object the environment uses
# =============================================================================
class PhysicsProvider:
    """Precomputes and caches per-(image, landmark) physics fields.

    Designed to be created once per environment and queried each step. All heavy
    computation happens on the first touch of a given (image, landmark) pair.
    """

    def __init__(self, beta=8.0, sigma=1.0, cache_dir=None, channels=("mri", "grad", "potential", "dist")):
        self.beta = beta
        self.sigma = sigma
        self.cache_dir = cache_dir
        self.channels = tuple(channels)
        self._mem = {}  # in-process cache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ---- keys / disk cache ------------------------------------------------
    def _key(self, image_id, seed_xyz):
        h = hashlib.md5(f"{image_id}|{seed_xyz}|{self.beta}|{self.sigma}".encode()).hexdigest()[:16]
        return h

    def _load_disk(self, key):
        if not self.cache_dir:
            return None
        p = os.path.join(self.cache_dir, key + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            return {k: z[k] for k in z.files}
        return None

    def _save_disk(self, key, fields):
        if not self.cache_dir:
            return
        np.savez_compressed(os.path.join(self.cache_dir, key + ".npz"), **fields)

    # ---- main API ---------------------------------------------------------
    def fields_for(self, volume, seed_xyz, image_id="img"):
        """Return dict with 'V' (geodesic potential), 'F' (speed), 'd' (euclid)."""
        key = self._key(image_id, tuple(map(int, seed_xyz)))
        if key in self._mem:
            return self._mem[key]
        disk = self._load_disk(key)
        if disk is not None:
            self._mem[key] = disk
            return disk
        F = speed_field(volume, self.beta, self.sigma)
        V = geodesic_potential(volume, seed_xyz, self.beta, self.sigma)
        d = euclidean_potential(volume.shape, seed_xyz)
        fields = {"V": V, "F": F, "d": d}
        self._save_disk(key, fields)
        self._mem[key] = fields
        return fields

    def channel_stack(self, volume, seed_xyz, image_id="img"):
        """Build the multi-channel physics tensor requested in Step 1 of the PDF.

        Returns an array of shape (C, X, Y, Z), normalized to [0,1] per channel.
        Redundant channels (e.g. grad vs potential) are correlated ~0.9; keep the
        set small for real training. Default: MRI, grad, potential, dist.
        """
        f = self.fields_for(volume, seed_xyz, image_id)
        maps = {
            "mri": _normalize(volume),
            "grad": _norm01(edge_strength(volume, self.sigma)),
            "vesselness": _norm01(hessian_vesselness(volume)),
            "potential": _norm01(f["V"]),
            "dist": _norm01(f["d"]),
            "speed": f["F"],
        }
        return np.stack([maps[c] for c in self.channels], axis=0).astype(np.float32)

    # ---- reward shaping ----------------------------------------------------
    def shaping_potential(self, volume, seed_xyz, loc, image_id="img"):
        """Phi(s) = -V(loc): the potential for Ng-1999 potential-based shaping.

        The shaped reward is   r' = r + gamma*Phi(s') - Phi(s),
        which is guaranteed to preserve the optimal policy while densifying the
        signal. Use this INSTEAD of adding raw ||grad V|| to the reward.
        """
        V = self.fields_for(volume, seed_xyz, image_id)["V"]
        i = tuple(np.clip(np.round(loc).astype(int), 0, np.array(V.shape) - 1))
        return -float(V[i])


def _norm01(a):
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


# =============================================================================
# Self-test (runs on any 3D numpy volume)
# =============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vol = np.load(sys.argv[1])
    else:
        # synthetic blob volume
        z, y, x = np.mgrid[0:60, 0:60, 0:60]
        vol = (np.sin(x / 5.0) * np.cos(y / 5.0) * 100 + 200).astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    seed = tuple(np.array(vol.shape) // 2)
    pp = PhysicsProvider()
    f = pp.fields_for(vol, seed)
    print("V range", float(f["V"].min()), float(f["V"].max()))
    print("F range", float(f["F"].min()), float(f["F"].max()))
    stk = pp.channel_stack(vol, seed)
    print("channel stack", stk.shape)
    print("Phi at a random loc:", pp.shaping_potential(vol, seed, [10, 10, 10]))
    print("physics.py self-test OK")
