"""Minimal NIfTI-1 reader (.nii / .nii.gz), dependency-free."""
import gzip
import struct
import numpy as np

_DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32,
       64: np.float64, 256: np.int8, 512: np.uint16}


def read_nii(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    assert struct.unpack("<i", raw[0:4])[0] == 348, "not a NIfTI-1 file"
    dim = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    pixdim = struct.unpack("<8f", raw[76:108])
    vox_offset = int(struct.unpack("<f", raw[108:112])[0])
    ndim = dim[0]
    shape = dim[1:1 + ndim]
    data = np.frombuffer(raw[vox_offset:], dtype=_DT[datatype])
    n = int(np.prod(shape))
    data = data[:n].reshape(shape, order="F").astype(np.float32)
    return data, pixdim[1:1 + ndim]


if __name__ == "__main__":
    import sys
    v, sp = read_nii(sys.argv[1])
    print("shape", v.shape, "spacing", sp, "range", float(v.min()), float(v.max()))
