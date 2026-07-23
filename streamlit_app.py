"""
RL-Medical / physics-informed landmark detector — Streamlit version.

Loads the bundled single-agent DQN checkpoint (trained to find landmark #13 -
the anterior commissure - in brain MRI) and runs it in inference-only
("play") mode on a NIfTI/DICOM volume the user uploads.

This wraps the existing CLI (`src/DQN.py --task play`) as a subprocess so the
original, unmodified training/inference code is reused as-is. The
input-handling logic (is_dicom / is_nifti / prepare_image) is unchanged from
the Gradio version on purpose.
"""

import ast
import base64
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
MODEL_PATH = os.path.join(SRC_DIR, "data", "models", "BrainMRI", "SingleAgent.pt")
LANDMARK_ID = 13  # what SingleAgent.pt was trained on
MAX_GIF_FRAMES = 60


# --------------------------------------------------------------------------
# Backend (unchanged logic from the Gradio app)
# --------------------------------------------------------------------------

def run_inference(nifti_path: str):
    """Runs `DQN.py --task play` on a single volume and returns the row
    printed by the logger with the agent's final (x, y, z) voxel position,
    plus the full step-by-step trajectory (from the STEP_LOC log lines)."""

    with tempfile.TemporaryDirectory() as tmp:
        files_list = os.path.join(tmp, "image_files.txt")
        with open(files_list, "w") as f:
            f.write(nifti_path + "\n")

        cmd = [
            sys.executable, "DQN.py",
            "--task", "play",
            "--load", MODEL_PATH,
            "--files", files_list,
            "--file_type", "brain",
            "--landmarks", str(LANDMARK_ID),
            "--model_name", "Network3d",
            "--viz", "0",
        ]
        result = subprocess.run(
            cmd, cwd=SRC_DIR, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Inference failed:\n" + result.stdout[-2000:] + "\n" + result.stderr[-2000:]
            )

        row = None
        trajectory = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("STEP_LOC:"):
                parts = line.split()
                try:
                    step, x, y, z = int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    trajectory.append((step, x, y, z))
                except (ValueError, IndexError):
                    continue
            elif line.startswith("[") and line.endswith("]"):
                try:
                    parsed = ast.literal_eval(line)
                    if isinstance(parsed, list) and isinstance(parsed[0], int):
                        row = parsed
                except (ValueError, SyntaxError):
                    continue
        if row is None:
            raise RuntimeError("Could not parse model output:\n" + result.stdout[-2000:])

        x, y, z = row[2], row[3], row[4]
        return int(round(x)), int(round(y)), int(round(z)), trajectory


def detect_format(path):
    """
    Detects the file type by reading the file contents,
    not by checking the filename extension.
    """

    # ---------- Try DICOM ----------
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
        return "dicom"
    except Exception:
        pass

    # ---------- Try NIfTI ----------
    try:
        nib.load(path)
        return "nifti"
    except Exception:
        pass

    return None


def prepare_image(path):

    try:
        nib.load(path)
        print("Detected NIfTI")
        return path
    except Exception:
        pass

    try:
        image = sitk.ReadImage(path)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".nii.gz",
            delete=False
        )

        sitk.WriteImage(image, tmp.name)

        print("Detected DICOM")

        return tmp.name

    except Exception:
        pass

    raise ValueError("Unsupported medical image format.")

def make_preview(nifti_path: str, x: int, y: int, z: int):
    image = sitk.ReadImage(nifti_path)
    array = sitk.GetArrayFromImage(image)
    zmax, ymax, xmax = array.shape
    x = min(max(x, 0), xmax - 1)
    y = min(max(y, 0), ymax - 1)
    z = min(max(z, 0), zmax - 1)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    views = [
        (array[z, :, :], (x, y), "Axial"),
        (array[:, y, :], (x, z), "Coronal"),
        (array[:, :, x], (y, z), "Sagittal"),
    ]
    for ax, (slice_, point, title) in zip(axs, views):
        ax.imshow(slice_, cmap="gray")
        ax.scatter([point[0]], [point[1]], c="red", s=40, marker="+")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def make_trajectory_gif(nifti_path: str, trajectory, out_path: str, fps: int = 8):
    if not trajectory:
        return None

    image = sitk.ReadImage(nifti_path)
    array = sitk.GetArrayFromImage(image)
    zmax, ymax, xmax = array.shape
    vmin, vmax = np.percentile(array, 2), np.percentile(array, 99)

    if len(trajectory) > MAX_GIF_FRAMES:
        idxs = np.linspace(0, len(trajectory) - 2, MAX_GIF_FRAMES - 1).astype(int)
        frames = [trajectory[i] for i in idxs] + [trajectory[-1]]
    else:
        frames = trajectory

    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="black")
    roi = 40
    xs_trail, ys_trail = [], []

    def render(frame):
        ax.clear()
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        step, x, y, z = frame
        x = int(min(max(x, 0), xmax - 1))
        y = int(min(max(y, 0), ymax - 1))
        z = int(min(max(z, 0), zmax - 1))
        ax.imshow(array[z, :, :], cmap="gray", vmin=vmin, vmax=vmax)
        xs_trail.append(x)
        ys_trail.append(y)
        ax.plot(xs_trail, ys_trail, "-", color="#7cf6ff", lw=1.6, alpha=0.9)
        ax.add_patch(plt.Rectangle((x - roi, y - roi), 2 * roi, 2 * roi,
                                    fill=False, edgecolor="#ffcc00", lw=1.4))
        ax.plot(x, y, "o", color="#3a7bff", ms=7, mec="w", mew=0.6)
        ax.set_title(f"Step {step}", color="#ffcc00", fontsize=10)
        fig.suptitle("Agent search path", color="white", fontsize=11)

    anim = FuncAnimation(fig, render, frames=frames)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return out_path


def gif_as_html(gif_path, width=420):
    """Streamlit's st.image doesn't animate gifs reliably in every version,
    so embed it directly as an <img> tag instead."""
    with open(gif_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/gif;base64,{data}" width="{width}">'


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Communicative RL Medical Imaging",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align:center;'>🧠 Communicative Reinforcement Learning Medical Imaging Platform</h1>
    <h4 style='text-align:center; font-weight:normal;'>Physics-Informed Landmark Detection for Brain MRI</h4>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "Upload either a **NIfTI** (`.nii` / `.nii.gz`) or a **DICOM** (`.dcm`) file. "
    "The AI automatically detects anatomical landmark #{}.".format(LANDMARK_ID)
)

col_upload, col_results = st.columns([1, 2])

with col_upload:
    st.subheader("📤 Upload")
    uploaded_files = st.file_uploader(
    "Upload MRI Images",
    accept_multiple_files=True,
    help="You can upload one or many NIfTI/DICOM images."
)
    run_button = st.button("🚀 Start AI Analysis", type="primary", use_container_width=True)
    metadata_box = st.empty()

# Image processing pipeline
if run_button:

    if not uploaded_files:
        st.error("Upload one or more MRI files first.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    top1, top2, top3 = st.columns(3)
    top1.metric(
    "Uploaded",
    len(uploaded_files)
    )
    top2.metric(
        "Completed",
        0
    )
    
    top3.metric(
        "Failed",
        0
    )

    successful = sum(
    1 for r in results
    if "Error" not in r
)

failed = len(results) - successful

top2.metric(
    "Completed",
    successful
)

top3.metric(
    "Failed",
    failed
)

    results = []

    with st.spinner("Running AI analysis..."):

        for idx, uploaded_file in enumerate(uploaded_files):

            status.info(
                f"Analyzing ({idx+1}/{len(uploaded_files)}): {uploaded_file.name}"
            )

            filename = uploaded_file.name.lower()

            if filename.endswith(".nii.gz"):
                suffix = ".nii.gz"
            elif filename.endswith(".nii"):
                suffix = ".nii"
            else:
                suffix = ""

            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            )

            tmp.write(uploaded_file.getbuffer())
            tmp.close()

            try:

                image_path = prepare_image(tmp.name)

                x, y, z, trajectory = run_inference(image_path)

                fig = make_preview(image_path, x, y, z)

                gif_fd, gif_path = tempfile.mkstemp(suffix=".gif")
                os.close(gif_fd)

                gif_path = make_trajectory_gif(
                    image_path,
                    trajectory,
                    gif_path
                )

                # ---------- Metadata ----------
                try:
                    img = nib.load(image_path)
                    arr = img.get_fdata()

                    info = (
                        f"Shape : {arr.shape}\n"
                        f"Data Type : {arr.dtype}\n"
                        f"Minimum : {arr.min():.2f}\n"
                        f"Maximum : {arr.max():.2f}"
                    )

                except Exception:

                    info = "Metadata unavailable."

                report_text = f"""
Filename : {uploaded_file.name}

Landmark : {LANDMARK_ID}

X : {x}
Y : {y}
Z : {z}

Steps : {trajectory[-1][0] if trajectory else 'N/A'}
"""

                # Save result
                results.append({
                    "Filename": uploaded_file.name,
                    "Landmark": LANDMARK_ID,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "Steps": trajectory[-1][0] if trajectory else "N/A"
                })

                progress.progress((idx + 1) / len(uploaded_files))

                # Display each result
                with st.expander(f"📁 {uploaded_file.name}", expanded=False):

                    st.pyplot(fig)

                    if gif_path:
                        st.markdown(
                            gif_as_html(gif_path),
                            unsafe_allow_html=True
                        )

                    st.text(report_text)

                    st.text(info)

            except Exception as e:

                results.append({
                    "Filename": uploaded_file.name,
                    "Landmark": "",
                    "X": "",
                    "Y": "",
                    "Z": "",
                    "Steps": "",
                    "Error": str(e)
                })

                st.error(
                    f"{uploaded_file.name} failed:\n{e}"
                )

    # ---------------- Summary ----------------

    import pandas as pd

    df = pd.DataFrame(results)

    st.success(
        f"Finished analysing {len(results)} image(s)."
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download CSV Report",
        csv,
        file_name="Landmark_Report.csv",
        mime="text/csv"
    )

# Display each patient's results
with st.expander(
    f"🧠 {uploaded_file.name}",
    expanded=(idx == 0)
):

    left, right = st.columns([2, 1])

    with left:

        st.subheader("MRI Visualization")

        st.pyplot(fig)

        if gif_path:

            st.subheader("RL Agent Search")

            st.markdown(
                gif_as_html(gif_path),
                unsafe_allow_html=True
            )

    with right:

        st.subheader("Detection Summary")

        st.success("Landmark Successfully Detected")

        c1, c2, c3 = st.columns(3)

        c1.metric("X", x)
        c2.metric("Y", y)
        c3.metric("Z", z)

        st.metric(
            "Landmark",
            f"#{LANDMARK_ID}"
        )

        st.divider()

        st.subheader("Image Information")

        st.code(info)

        st.download_button(
            "📄 Download Report",
            report_text,
            file_name=f"{uploaded_file.name}_report.txt",
            mime="text/plain",
            key=f"report_{idx}"
        )
