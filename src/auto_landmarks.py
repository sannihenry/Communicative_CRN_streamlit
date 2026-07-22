#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: auto_landmarks.py
#
# Generate ANATOMICALLY-DEFINED landmarks automatically from a brain volume.
#
# HONEST SCOPE ---------------------------------------------------------------
# These are REPRODUCIBLE, ANATOMICALLY-NAMED GEOMETRIC FIDUCIALS derived from the
# brain mask + the scan's true patient orientation (from DICOM). They land in
# sensible anatomical places (cranial vertex, frontal/occipital poles, temporal
# extremes, midsagittal centre, an AC-PC-ish midline set). They are suitable for:
#   * driving the physics / geodesic navigation,
#   * bootstrapping / pre-training,
#   * sanity-checking the pipeline.
# They are NOT a substitute for expert AC/PC-type annotation or atlas-propagated
# labels. For a clinical detector use one of the two routes in AUTO_LANDMARKS.md
# (atlas registration, or manual annotation -> annotations_to_csv.py).
# ----------------------------------------------------------------------------
#
# The key idea: we don't take raw voxel-axis extremes (wrong when the scan is
# tilted). We project each brain voxel onto the TRUE patient L/P/S directions
# (from ImageOrientationPatient) and take extremes in anatomical space.

import numpy as np
from scipy import ndimage as ndi


def brain_mask(vol, pct=55):
    """Largest connected, hole-filled bright component (a crude brain/head mask)."""
    thr = np.percentile(vol, pct)
    m = vol > thr
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n == 0:
        return np.ones_like(vol, dtype=bool)
    sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    return m


def patient_axes(cosines):
    """cosines = (row_dir, col_dir) each length-3 (DICOM ImageOrientationPatient).
    Returns L, P, S unit direction vectors in *voxel-axis* space (index 0,1,2 = x,y,z).

    In LPS patient space, +x=Left, +y=Posterior, +z=Superior. The patient
    coordinate of voxel (x,y,z) is ~ x*sx*row + y*sy*col + z*sz*normal. So the
    'how-Left/Posterior/Superior' score of a voxel is the projection of its
    (scaled) index onto the row/col/normal component for that patient axis.
    """
    row = np.asarray(cosines[0], float)
    col = np.asarray(cosines[1], float)
    nrm = np.cross(row, col)
    M = np.stack([row, col, nrm], axis=1)   # columns: contribution of each voxel axis
    # M[patient_axis, voxel_axis]; row 0->Left(+x), 1->Posterior(+y), 2->Superior(+z)
    L = M[0]; P = M[1]; S = M[2]
    return L, P, S


def anatomical_fiducials(vol, cosines=None, spacing=(1, 1, 1), n=15):
    """Return (names, pts[n,3] voxel indices) of anatomical fiducials."""
    vol = np.asarray(vol, float)
    if vol.ndim == 4:
        vol = vol[..., 0]
    m = brain_mask(vol)
    xs, ys, zs = np.where(m)
    idx = np.stack([xs, ys, zs], 1).astype(float)          # (N,3) voxel indices
    scaled = idx * np.asarray(spacing, float)              # physical-ish
    centroid = idx.mean(0)

    if cosines is None:                                    # assume axial identity
        L = np.array([1., 0, 0]); P = np.array([0, 1., 0]); S = np.array([0, 0, 1.])
    else:
        L, P, S = patient_axes(cosines)

    def extreme(direction, most=True):
        proj = scaled @ np.asarray(direction, float)
        j = int(np.argmax(proj) if most else np.argmin(proj))
        return idx[j]

    fid = {
        "centroid":        centroid,
        "cranial_vertex":  extreme(S, True),      # most superior
        "inferior":        extreme(S, False),     # most inferior
        "frontal_pole":    extreme(P, False),     # most anterior (min Posterior)
        "occipital_pole":  extreme(P, True),      # most posterior
        "left_extreme":    extreme(L, True),      # most left
        "right_extreme":   extreme(L, False),     # most right
    }

    # midsagittal centre: centroid pulled onto the L-R midplane (mean L-projection)
    Lp = scaled @ L
    mid_val = (Lp.min() + Lp.max()) / 2.0
    # closest brain voxel to (mid L-plane, centroid P & S)
    Pp = scaled @ P; Sp = scaled @ S
    cP = (idx @ np.eye(3))  # placeholder
    cscaled = centroid * np.asarray(spacing, float)
    dist = (Lp - mid_val) ** 2 + (Pp - cscaled @ P) ** 2 + (Sp - cscaled @ S) ** 2
    fid["midsagittal_center"] = idx[int(np.argmin(dist))]

    # AC-PC-ish midline series: along the P axis on the midsagittal plane at
    # centroid S-level, sample a few interior points (anterior->posterior).
    near_mid = np.abs(Lp - mid_val) < (0.06 * (Lp.max() - Lp.min()) + 1e-6)
    near_S = np.abs(Sp - cscaled @ S) < (0.10 * (Sp.max() - Sp.min()) + 1e-6)
    band = near_mid & near_S
    names = list(fid.keys())
    pts = list(fid.values())
    if band.sum() >= 4:
        bidx = idx[band]; bP = (bidx * np.asarray(spacing, float)) @ P
        order = np.argsort(bP)
        picks = np.linspace(0, len(order) - 1, 6).astype(int)
        for r, pk in enumerate(picks):
            names.append(f"midline_ap_{r}")
            pts.append(bidx[order[pk]])

    # pad/truncate to exactly n, filling extra slots with interpolations toward centroid
    while len(pts) < n:
        a = pts[len(pts) % 7]
        names.append(f"aux_{len(pts)}")
        pts.append((np.asarray(a) + centroid) / 2.0)
    names = names[:n]; pts = np.array(pts[:n])
    pts = np.clip(np.round(pts), 0, np.asarray(vol.shape) - 1).astype(int)
    return names, pts


def write_landmark_file(path, pts):
    with open(path, "w") as f:
        f.write("\n".join(f"{int(x)},{int(y)},{int(z)}" for x, y, z in pts) + "\n")


if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from read_nifti import read_nii
    ap = argparse.ArgumentParser(description="Auto anatomical fiducials for a NIfTI volume.")
    ap.add_argument("nii")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    v, sp = read_nii(a.nii)
    names, pts = anatomical_fiducials(v, cosines=None, spacing=sp, n=a.n)
    out = a.out or (os.path.splitext(a.nii)[0] + "_landmarks.txt")
    write_landmark_file(out, pts)
    for nm, p in zip(names, pts):
        print(f"  {nm:20s} {tuple(int(x) for x in p)}")
    print("wrote", out)
