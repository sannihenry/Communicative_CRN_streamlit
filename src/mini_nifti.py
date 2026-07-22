"""Minimal NIfTI-1 writer (.nii.gz), dependency-free. Fallback for when
SimpleITK/nibabel are unavailable. Pairs with read_nifti.read_nii."""
import gzip
import struct
import numpy as np


def write_nii(path, vol, spacing=(1.0, 1.0, 1.0)):
    vol = np.asarray(vol, dtype=np.float32)
    assert vol.ndim == 3, "expected a 3D volume"
    nx, ny, nz = vol.shape
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)                       # sizeof_hdr
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)  # dim
    struct.pack_into("<h", hdr, 70, 16)                       # datatype = FLOAT32
    struct.pack_into("<h", hdr, 72, 32)                       # bitpix
    struct.pack_into("<8f", hdr, 76, 0.0, float(spacing[0]), float(spacing[1]),
                     float(spacing[2]), 1.0, 1.0, 1.0, 1.0)   # pixdim
    struct.pack_into("<f", hdr, 108, 352.0)                   # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)                     # scl_slope
    struct.pack_into("<h", hdr, 252, 1)                       # qform_code
    struct.pack_into("<h", hdr, 254, 1)                       # sform_code
    # simple diagonal sform
    struct.pack_into("<4f", hdr, 280, float(spacing[0]), 0, 0, 0)
    struct.pack_into("<4f", hdr, 296, 0, float(spacing[1]), 0, 0)
    struct.pack_into("<4f", hdr, 312, 0, 0, float(spacing[2]), 0)
    hdr[344:348] = b"n+1\x00"                                 # magic
    data = np.ascontiguousarray(vol.transpose(2, 1, 0)).astype("<f4").tobytes(order="C")
    # NIfTI stores fastest-varying first (x), so write in Fortran order of (nx,ny,nz)
    data = np.asfortranarray(vol).astype("<f4").tobytes(order="F")
    with gzip.open(path, "wb") as f:
        f.write(bytes(hdr) + b"\x00\x00\x00\x00" + data)
