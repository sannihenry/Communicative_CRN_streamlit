#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: prepare_data.py
#
# Turn an arbitrary tree of DICOM data into the folder layout the RL-Medical /
# PINN pipeline expects:
#
#     data/
#       <anything>/.../<dicom slices>     <-- your raw data, ANY structure
#       images/        <-- generated: one <name>.nii.gz per DICOM series
#       filenames/     <-- generated: image_files.txt, landmark_files.txt
#       landmarks/     <-- generated: one <name>.txt per volume (>=15 points)
#       models/        <-- (left untouched)
#
# It walks everything under --data recursively, groups DICOM files into series
# (3D volumes), converts each to .nii.gz, and writes the filenames + landmark
# files so `dataReader.py` / `medical.py` run unchanged.
#
# Backends (auto-detected, best first):
#   1. SimpleITK  -- robust series grouping + ordering (recommended; the repo
#                    already depends on it).
#   2. pydicom (+ mini_nifti writer)
#   3. mini_dicom (built-in, dependency-free fallback)
#
# LANDMARKS: you almost certainly do NOT have annotations yet. This script
# writes VALID PLACEHOLDER landmark files (>=15 points, correct format) so the
# pipeline is runnable immediately, and prints a loud warning. Replace them with
# real annotations before training a detector, or pass --landmarks-csv to fill
# in known points. (For the physics/PINN visualisations, the placeholder seed is
# enough to compute a geodesic field.)
#
# Usage:
#   python prepare_data.py --data /path/to/data
#   python prepare_data.py --data ./data --num-landmarks 15 --landmarks centroid
#   python prepare_data.py --data ./data --landmarks-csv my_points.csv
#
# CSV format for --landmarks-csv (one row per point):
#   volume_name,index,x,y,z
#   2_MR5OOO_20250514_1,13,89,88,84      # AC point for that volume
#   2_MR5OOO_20250514_1,14,88,93,77      # PC point

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- optional backends ------------------------------------------------------
try:
    import SimpleITK as sitk
    _HAVE_SITK = True
except Exception:
    _HAVE_SITK = False
try:
    import pydicom
    _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

import mini_dicom
from mini_nifti import write_nii


# =============================================================================
# Series discovery -- works on ANY folder structure
# =============================================================================
def find_dicom_dirs(root, images_dirname):
    """Yield directories under `root` that contain at least one DICOM file,
    skipping the generated output folders."""
    skip = {images_dirname, "filenames", "landmarks", "models"}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = set(rel.split(os.sep))
        if parts & skip:
            continue
        dcm = [os.path.join(dirpath, f) for f in filenames
               if _looks_dicom(os.path.join(dirpath, f))]
        if dcm:
            yield dirpath, dcm


def _looks_dicom(path):
    if not os.path.isfile(path):
        return False
    low = path.lower()
    if low.endswith((".dcm", ".ima")):
        return True
    if low.endswith((".nii", ".nii.gz", ".txt", ".json", ".png", ".gif", ".pt")):
        return False
    return mini_dicom.is_dicom(path)   # DICM magic at byte 128 (extension-less ok)


def group_series_sitk(dirpath):
    """Return {series_uid: [ordered file paths]} using GDCM (robust)."""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(dirpath)
    out = {}
    if ids:
        for sid in ids:
            files = reader.GetGDCMSeriesFileNames(dirpath, sid)
            if files:
                out[sid] = list(files)
    return out


def group_series_fallback(dcm_files):
    """Group by SeriesInstanceUID via mini_dicom; if unreadable, one series/dir."""
    groups = {}
    for p in dcm_files:
        try:
            d = mini_dicom.read_dicom(p)
            sid = d.get("SeriesInstanceUID") or "series"
        except Exception:
            sid = "series"
        groups.setdefault(sid, []).append(p)
    return groups


# =============================================================================
# Reading a series into a volume + spacing
# =============================================================================
def load_series(files, dirpath):
    if _HAVE_SITK:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        img = reader.Execute()
        vol = sitk.GetArrayFromImage(img)          # (z, y, x)
        vol = np.transpose(vol, (2, 1, 0)).astype(np.float32)   # -> (x, y, z)
        return vol, tuple(img.GetSpacing())        # (sx, sy, sz)
    if _HAVE_PYDICOM:
        slices = []
        for p in files:
            ds = pydicom.dcmread(p)
            slices.append(ds)
        slices.sort(key=lambda s: float(getattr(s, "InstanceNumber", 0)))
        vol = np.stack([s.pixel_array for s in slices], axis=-1)  # (rows,cols,z)=(y,x,z)
        vol = np.transpose(vol, (1, 0, 2)).astype(np.float32)     # -> (x,y,z)
        ps = [float(x) for x in getattr(slices[0], "PixelSpacing", [1, 1])]
        st = float(getattr(slices[0], "SliceThickness", 1) or 1)
        return vol, (ps[1], ps[0], st)
    return mini_dicom.volume_from_series(files)


# =============================================================================
# Landmark file generation
# =============================================================================
def make_landmarks(vol, mode, n, provided):
    """Return an (n,3) array of landmark voxel coords.

    provided: {index: (x,y,z)} of known points for THIS volume (from CSV).
    Unknown indices are filled with a placeholder (center or intensity centroid).
    """
    if mode == "centroid":
        thr = np.percentile(vol, 60)
        mask = vol > thr
        if mask.any():
            xs, ys, zs = np.where(mask)
            base = np.array([xs.mean(), ys.mean(), zs.mean()])
        else:
            base = np.array(vol.shape) / 2.0
    else:  # center
        base = np.array(vol.shape) / 2.0
    pts = np.tile(base, (n, 1))
    # scatter placeholders slightly so they're not all identical
    offs = np.linspace(-6, 6, n)
    pts[:, 0] += offs
    for idx, xyz in (provided or {}).items():
        if 0 <= idx < n:
            pts[idx] = xyz
    return np.clip(np.round(pts), 0, np.array(vol.shape) - 1).astype(int)


def _series_cosines(files):
    """Read ImageOrientationPatient (row_dir, col_dir) from the first slice.
    Returns None if unavailable (auto_landmarks then assumes axial identity)."""
    for p in files:
        try:
            if _HAVE_PYDICOM:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                iop = [float(x) for x in ds.ImageOrientationPatient]
            else:
                d = mini_dicom.read_dicom(p)
                iop = [float(x) for x in d["ImageOrientationPatient"].split("\\")]
            return (iop[:3], iop[3:])
        except Exception:
            continue
    return None


def load_csv(path):
    """Parse --landmarks-csv into {volume_name: {index: (x,y,z)}}."""
    table = {}
    if not path:
        return table
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("volume"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            name, idx, x, y, z = parts[0], int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            table.setdefault(name, {})[idx] = (x, y, z)
    return table


# =============================================================================
# Main
# =============================================================================
def sanitize(root, dirpath, series_key, used):
    """Build a clean, unique volume name from the path (structure-agnostic)."""
    rel = os.path.relpath(dirpath, root)
    parts = [p for p in rel.split(os.sep) if p and p.upper() != "DICOM"]
    name = "_".join(parts) if parts else "vol"
    # disambiguate multiple series in the same dir
    if series_key and series_key != "series":
        name += "_s" + series_key.split(".")[-1][-4:]
    name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
    base = name
    k = 1
    while name in used:
        name = f"{base}_{k}"; k += 1
    used.add(name)
    return name


def main():
    ap = argparse.ArgumentParser(description="Prepare DICOM data for RL-Medical/PINN.")
    ap.add_argument("--data", required=True, help="path to the top-level data/ folder")
    ap.add_argument("--images-dir", default="images", help="output folder for .nii.gz")
    ap.add_argument("--num-landmarks", type=int, default=15,
                    help=">=15 required (indices 13=AC, 14=PC are used by the code)")
    ap.add_argument("--landmarks", choices=["center", "centroid", "auto"], default="center",
                    help="center/centroid = placeholder; auto = anatomical fiducials "
                         "(oriented by the DICOM header; see auto_landmarks.py / AUTO_LANDMARKS.md)")
    ap.add_argument("--landmarks-csv", default=None, help="known points to fill in")
    ap.add_argument("--min-slices", type=int, default=3,
                    help="ignore series with fewer slices than this")
    ap.add_argument("--relative", action="store_true",
                    help="write relative paths in filename lists (default: absolute)")
    ap.add_argument("--overwrite", action="store_true",
                    help="reconvert volumes even if the .nii.gz already exists")
    args = ap.parse_args()

    root = os.path.abspath(args.data)
    img_out = os.path.join(root, args.images_dir)
    fn_out = os.path.join(root, "filenames")
    lm_out = os.path.join(root, "landmarks")
    for d in (img_out, fn_out, lm_out):
        os.makedirs(d, exist_ok=True)

    backend = "SimpleITK" if _HAVE_SITK else ("pydicom" if _HAVE_PYDICOM else "mini_dicom (built-in)")
    print(f"[prepare_data] backend: {backend}")
    print(f"[prepare_data] scanning: {root}")

    csv_points = load_csv(args.landmarks_csv)
    if args.num_landmarks < 15:
        print("[warn] num-landmarks < 15; the code indexes landmark 13/14 (AC/PC). Forcing 15.")
        args.num_landmarks = 15

    used_names = set()
    image_lines, landmark_lines = [], []
    n_series = 0

    for dirpath, dcm_files in find_dicom_dirs(root, args.images_dir):
        if _HAVE_SITK:
            groups = group_series_sitk(dirpath) or group_series_fallback(dcm_files)
        else:
            groups = group_series_fallback(dcm_files)

        for series_key, files in groups.items():
            if len(files) < args.min_slices:
                continue
            name = sanitize(root, dirpath, series_key, used_names)
            nii_path = os.path.join(img_out, name + ".nii.gz")
            try:
                if args.overwrite or not os.path.exists(nii_path):
                    vol, spacing = load_series(files, dirpath)
                    if not _HAVE_SITK:
                        write_nii(nii_path, vol, spacing)
                    else:
                        reader = sitk.ImageSeriesReader(); reader.SetFileNames(files)
                        sitk.WriteImage(reader.Execute(), nii_path)
                        vol, spacing = load_series(files, dirpath)  # for landmark shape
                else:
                    from read_nifti import read_nii
                    vol, spacing = read_nii(nii_path)
                    if vol.ndim == 4:
                        vol = vol[..., 0]
            except Exception as e:
                print(f"[skip] {dirpath} [{series_key[:12]}]: {e}")
                continue

            # landmark file
            if args.landmarks == "auto":
                from auto_landmarks import anatomical_fiducials
                cos = _series_cosines(files)
                _, lm = anatomical_fiducials(vol, cosines=cos, spacing=spacing,
                                             n=args.num_landmarks)
                # let CSV points override specific indices if provided
                for idx, xyz in (csv_points.get(name) or {}).items():
                    if 0 <= idx < len(lm):
                        lm[idx] = np.clip(np.round(xyz), 0, np.array(vol.shape) - 1).astype(int)
            else:
                lm = make_landmarks(vol, args.landmarks, args.num_landmarks,
                                    csv_points.get(name))
            lm_path = os.path.join(lm_out, name + ".txt")
            with open(lm_path, "w") as f:
                f.write("\n".join(f"{int(x)},{int(y)},{int(z)}" for x, y, z in lm) + "\n")

            ip = nii_path if not args.relative else os.path.relpath(nii_path, root)
            lp = lm_path if not args.relative else os.path.relpath(lm_path, root)
            image_lines.append(ip)
            landmark_lines.append(lp)
            n_series += 1
            print(f"  + {name}  shape={tuple(vol.shape)}  slices={len(files)}")

    with open(os.path.join(fn_out, "image_files.txt"), "w") as f:
        f.write("\n".join(image_lines) + ("\n" if image_lines else ""))
    with open(os.path.join(fn_out, "landmark_files.txt"), "w") as f:
        f.write("\n".join(landmark_lines) + ("\n" if landmark_lines else ""))

    print(f"\n[prepare_data] done: {n_series} volume(s)")
    print(f"  images    -> {img_out}")
    print(f"  filenames -> {os.path.join(fn_out, 'image_files.txt')}  (+ landmark_files.txt)")
    print(f"  landmarks -> {lm_out}")
    have_real = bool(csv_points)
    if not have_real:
        print("\n" + "!" * 72)
        print("!  LANDMARKS ARE PLACEHOLDERS (image center/centroid), NOT real anatomy.")
        print("!  They let the pipeline + physics run, but you MUST replace them with")
        print("!  real annotations (or pass --landmarks-csv) before training a detector.")
        print("!" * 72)


if __name__ == "__main__":
    main()
