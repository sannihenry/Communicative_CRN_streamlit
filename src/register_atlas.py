#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: register_atlas.py
#
# Recommended production route for clinically-meaningful landmarks:
# register each subject to a TEMPLATE that already has expert landmark labels,
# then propagate those labels into the subject via the inverse transform.
#
# This yields consistent, atlas-defined anatomical points (e.g. AC/PC, commissures,
# ventricle horns) for every scan, without annotating each one by hand. Accuracy
# depends on registration quality; verify on a few cases.
#
# REQUIREMENTS (not run in this sandbox): SimpleITK, a template volume, and a
# template landmark file. Good public choices:
#   * MNI152 T1 template  (e.g. from templateflow / FSL) as the fixed image
#   * a landmark set defined once on that template (you place these ONCE)
#
# Usage:
#   python register_atlas.py \
#       --template MNI152_T1_1mm.nii.gz \
#       --template-landmarks mni_landmarks.txt \
#       --moving data/images/<subject>.nii.gz \
#       --out data/landmarks/<subject>.txt

import argparse
import numpy as np

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


def read_pts(path):
    return np.array([[float(v) for v in l.split(",")] for l in open(path) if l.strip()])


def register_and_propagate(template, moving, template_landmarks_vox):
    """Affine + B-spline register template->moving, map template landmark voxels
    into moving voxel space. Returns (n,3) voxel indices in the moving image."""
    fixed = sitk.ReadImage(moving, sitk.sitkFloat32)     # subject = fixed
    tmpl = sitk.ReadImage(template, sitk.sitkFloat32)    # atlas = moving-to-align

    # initialize + affine
    init = sitk.CenteredTransformInitializer(
        fixed, tmpl, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(1.0, 200)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    affine = reg.Execute(fixed, tmpl)

    # map each template landmark: voxel(tmpl) -> physical(tmpl) -> physical(fixed)
    # -> voxel(fixed). The registration aligns tmpl to fixed, so the transform maps
    # fixed-physical -> tmpl-physical; we need its inverse for tmpl-point -> fixed.
    inv = affine.GetInverse()
    out = []
    for vox in template_landmarks_vox:
        phys_t = tmpl.TransformContinuousIndexToPhysicalPoint([float(c) for c in vox])
        phys_f = inv.TransformPoint(phys_t)
        vox_f = fixed.TransformPhysicalPointToIndex(phys_f)
        out.append(tuple(int(v) for v in vox_f))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--template-landmarks", required=True,
                    help="voxel coords on the template, 'x,y,z' per line")
    ap.add_argument("--moving", required=True, help="subject .nii.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if sitk is None:
        raise SystemExit("SimpleITK is required: pip install SimpleITK")
    tl = read_pts(a.template_landmarks)
    pts = register_and_propagate(a.template, a.moving, tl)
    with open(a.out, "w") as f:
        f.write("\n".join(f"{x},{y},{z}" for x, y, z in pts) + "\n")
    print(f"propagated {len(pts)} landmarks -> {a.out}")


if __name__ == "__main__":
    main()
