"""Minimal, dependency-free DICOM I/O (explicit VR little-endian).

This is NOT a full DICOM implementation. It handles exactly the tags needed to
reconstruct a 3D MR volume: geometry, ordering, and 16-bit pixel data. It exists
so (a) the pipeline can be tested without pydicom/SimpleITK, and (b) it can serve
as a last-resort fallback backend in prepare_data.py. In production you should
prefer SimpleITK or pydicom (auto-detected there).
"""
import struct
import numpy as np

# tags we care about  (group, element) -> name
TAGS = {
    (0x0008, 0x0060): "Modality",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0013): "InstanceNumber",
    (0x0020, 0x0032): "ImagePositionPatient",
    (0x0020, 0x0037): "ImageOrientationPatient",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0030): "PixelSpacing",
    (0x0018, 0x0050): "SliceThickness",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0101): "BitsStored",
    (0x0028, 0x0103): "PixelRepresentation",
    (0x7FE0, 0x0010): "PixelData",
}
_EXPLICIT_LONG_VR = {"OB", "OW", "OF", "SQ", "UT", "UN"}


def is_dicom(path):
    """True if the file has the DICM magic at byte 128 (works for extension-less files)."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def read_dicom(path):
    """Parse a single explicit-VR-LE DICOM file into a dict of the TAGS above."""
    with open(path, "rb") as f:
        buf = f.read()
    if buf[128:132] != b"DICM":
        raise ValueError("not a preambled DICOM: " + path)
    i = 132
    out = {}
    n = len(buf)
    while i + 8 <= n:
        group, elem = struct.unpack_from("<HH", buf, i); i += 4
        vr = buf[i:i + 2].decode("ascii", "ignore"); i += 2
        if vr in _EXPLICIT_LONG_VR:
            i += 2  # reserved
            (length,) = struct.unpack_from("<I", buf, i); i += 4
        elif vr.isalpha() and vr.isupper() and len(vr) == 2:
            (length,) = struct.unpack_from("<H", buf, i); i += 2
        else:
            # implicit VR fallback: the 2 bytes we read as VR are really length low half
            i -= 2
            (length,) = struct.unpack_from("<I", buf, i); i += 4
            vr = ""
        if length == 0xFFFFFFFF:  # undefined length (SQ) - skip, we don't need it
            length = 0
        val = buf[i:i + length]; i += length
        name = TAGS.get((group, elem))
        if name is None:
            continue
        if name == "PixelData":
            out["PixelData"] = val
        elif name in ("Rows", "Columns", "BitsAllocated", "BitsStored", "PixelRepresentation"):
            out[name] = struct.unpack_from("<H", val, 0)[0] if len(val) >= 2 else 0
        else:
            out[name] = val.decode("ascii", "ignore").strip("\x00 ")
    return out


def _ds_floats(s, k):
    try:
        return [float(x) for x in s.split("\\")][:k]
    except Exception:
        return [0.0] * k


def volume_from_series(paths):
    """Stack a list of single-slice DICOM paths into an ordered 3D array + spacing.

    Ordering: by projection onto the slice normal (from ImageOrientationPatient +
    ImagePositionPatient) when available, else by InstanceNumber, else filename.
    Returns (volume[X,Y,Z] float32, spacing=(sx,sy,sz)).
    """
    slices = []
    for p in paths:
        try:
            d = read_dicom(p)
            if "PixelData" in d and "Rows" in d:
                slices.append((p, d))
        except Exception:
            continue
    if not slices:
        raise ValueError("no readable slices")

    def sort_key(item):
        _, d = item
        ipp = _ds_floats(d.get("ImagePositionPatient", ""), 3)
        iop = _ds_floats(d.get("ImageOrientationPatient", ""), 6)
        if len(ipp) == 3 and len(iop) == 6:
            r = np.array(iop[:3]); c = np.array(iop[3:]); nrm = np.cross(r, c)
            return float(np.dot(nrm, ipp))
        try:
            return float(d.get("InstanceNumber", 0))
        except Exception:
            return 0.0

    slices.sort(key=sort_key)
    rows = slices[0][1]["Rows"]; cols = slices[0][1]["Columns"]
    signed = slices[0][1].get("PixelRepresentation", 0) == 1
    dtype = np.int16 if signed else np.uint16
    planes = []
    for _, d in slices:
        arr = np.frombuffer(d["PixelData"], dtype=dtype)[: rows * cols].reshape(rows, cols)
        planes.append(arr.astype(np.float32))
    vol = np.stack(planes, axis=-1)                 # (rows, cols, nslices) = (Y, X, Z)
    vol = np.transpose(vol, (1, 0, 2))              # -> (X, Y, Z)
    ps = _ds_floats(slices[0][1].get("PixelSpacing", "1\\1"), 2) or [1.0, 1.0]
    # slice spacing from position delta if possible
    if len(slices) > 1:
        z0 = sort_key(slices[0]); z1 = sort_key(slices[-1])
        sz = abs(z1 - z0) / max(1, len(slices) - 1) or _ds_floats(slices[0][1].get("SliceThickness", "1"), 1)[0]
    else:
        sz = _ds_floats(slices[0][1].get("SliceThickness", "1"), 1)[0] or 1.0
    spacing = (float(ps[1]), float(ps[0]), float(sz))
    return vol, spacing


# --------------------------------------------------------------------------
# Minimal writer (used only by the test harness to synthesize DICOM series)
# --------------------------------------------------------------------------
def _elem(group, elem, vr, val):
    if isinstance(val, str):
        val = val.encode("ascii")
        if len(val) % 2:
            val += b" " if vr not in ("UI",) else b"\x00"
    head = struct.pack("<HH", group, elem) + vr.encode("ascii")
    if vr in _EXPLICIT_LONG_VR:
        head += b"\x00\x00" + struct.pack("<I", len(val))
    else:
        head += struct.pack("<H", len(val))
    return head + val


def write_dicom(path, arr2d, series_uid, instance, z, spacing=(1, 1, 1)):
    arr2d = arr2d.astype(np.uint16)
    rows, cols = arr2d.shape
    body = b"".join([
        _elem(0x0008, 0x0060, "CS", "MR"),
        _elem(0x0020, 0x000D, "UI", "1.2.3.4.5"),
        _elem(0x0020, 0x000E, "UI", series_uid),
        _elem(0x0020, 0x0013, "IS", str(instance)),
        _elem(0x0020, 0x0032, "DS", f"0\\0\\{z}"),
        _elem(0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0"),
        _elem(0x0028, 0x0010, "US", struct.pack("<H", rows)),
        _elem(0x0028, 0x0011, "US", struct.pack("<H", cols)),
        _elem(0x0028, 0x0030, "DS", f"{spacing[1]}\\{spacing[0]}"),
        _elem(0x0018, 0x0050, "DS", str(spacing[2])),
        _elem(0x0028, 0x0100, "US", struct.pack("<H", 16)),
        _elem(0x0028, 0x0101, "US", struct.pack("<H", 16)),
        _elem(0x0028, 0x0103, "US", struct.pack("<H", 0)),
        _elem(0x7FE0, 0x0010, "OW", arr2d.tobytes()),
    ])
    with open(path, "wb") as f:
        f.write(b"\x00" * 128 + b"DICM" + body)
