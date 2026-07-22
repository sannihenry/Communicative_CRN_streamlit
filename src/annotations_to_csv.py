#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: annotations_to_csv.py
#
# Convert manually-placed anatomical landmarks (the GOLD STANDARD) into the
# --landmarks-csv format that prepare_data.py consumes:
#
#     volume_name,index,x,y,z
#
# Supported inputs:
#   * 3D Slicer markups  .mrk.json  (Slicer 5+)     --format slicer-json
#   * 3D Slicer fiducials .fcsv                       --format fcsv
#   * ITK-SNAP label / simple "x y z name" text       --format xyz
#
# IMPORTANT: Slicer/ITK-SNAP store points in PHYSICAL (LPS/RAS) millimetre
# coordinates. The RL pipeline works in VOXEL indices. Pass the matching
# converted NIfTI with --nii so points are mapped physical -> voxel correctly
# (requires SimpleITK; without it, points are assumed already in voxel indices).
#
# Usage:
#   python annotations_to_csv.py --format slicer-json \
#       --in caseA.mrk.json --volume-name 3_MRI00019_20241125_15_s2458 \
#       --nii data/images/3_MRI00019_20241125_15_s2458.nii.gz --out points.csv

import argparse
import json
import os

try:
    import SimpleITK as sitk
    _HAVE_SITK = True
except Exception:
    _HAVE_SITK = False


def _phys_to_voxel(points_mm, nii):
    """Map physical LPS mm -> voxel index using the NIfTI geometry."""
    if not (nii and _HAVE_SITK and os.path.exists(nii)):
        return [tuple(round(c) for c in p) for p in points_mm]
    img = sitk.ReadImage(nii)
    out = []
    for p in points_mm:
        vox = img.TransformPhysicalPointToIndex([float(p[0]), float(p[1]), float(p[2])])
        out.append(tuple(int(v) for v in vox))
    return out


def read_slicer_json(path):
    data = json.load(open(path))
    pts = []
    for mk in data.get("markups", []):
        for cp in mk.get("controlPoints", []):
            pts.append(cp["position"])          # [x,y,z] in the file's coordinate system
    return pts


def read_fcsv(path):
    pts = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if len(f) >= 4:
            pts.append([float(f[1]), float(f[2]), float(f[3])])
    return pts


def read_xyz(path):
    pts = []
    for line in open(path):
        f = line.replace(",", " ").split()
        if len(f) >= 3:
            try:
                pts.append([float(f[0]), float(f[1]), float(f[2])])
            except ValueError:
                continue
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["slicer-json", "fcsv", "xyz"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--volume-name", required=True,
                    help="must match the .nii.gz basename produced by prepare_data.py")
    ap.add_argument("--nii", default=None, help="matching converted volume (for mm->voxel)")
    ap.add_argument("--out", default="points.csv")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()

    reader = {"slicer-json": read_slicer_json, "fcsv": read_fcsv, "xyz": read_xyz}[a.format]
    pts_mm = reader(a.inp)
    pts = _phys_to_voxel(pts_mm, a.nii)

    mode = "a" if a.append else "w"
    write_header = not (a.append and os.path.exists(a.out))
    with open(a.out, mode) as f:
        if write_header:
            f.write("volume_name,index,x,y,z\n")
        for i, (x, y, z) in enumerate(pts):
            f.write(f"{a.volume_name},{i},{int(x)},{int(y)},{int(z)}\n")
    print(f"wrote {len(pts)} points for {a.volume_name} -> {a.out}"
          + ("" if (a.nii and _HAVE_SITK) else "  [assumed voxel coords; pass --nii + SimpleITK for mm->voxel]"))


if __name__ == "__main__":
    main()
