"""
RL-Medical / physics-informed landmark detector — Streamlit version.

Loads one of the bundled DQN checkpoints (single-agent or multi-agent
CommNet/Network3d models trained on brain MRI landmarks) and runs it in
inference-only ("play") mode on NIfTI/DICOM volumes the user uploads.

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
import pandas as pd

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
MODELS_DIR = os.path.join(SRC_DIR, "data", "models", "BrainMRI")
MAX_GIF_FRAMES = 60

# --------------------------------------------------------------------------
# Available checkpoints
# --------------------------------------------------------------------------
# Landmark sets for the 5- and 8-agent CommNet models and the 8-agent
# Network3d model, plus the single-agent model, are documented in the
# upstream repo's README. The 3-agent landmark set is NOT documented
# upstream; it's assumed here to be the first three of the 5-agent set
# (13, 14, 0). If your CommNet3agents.pt was trained on a different set of
# landmarks, update the "landmarks" list below to match, or the model will
# load but predict against the wrong targets.
MODEL_CONFIGS = {
    "PINNfor1agent — (anterior commissure)": {
        "checkpoint": "PINNfor1agent.pt",
        "model_name": "Network3d",
        "landmarks": [13],
    },
    "PINNfor3agents": {
        "checkpoint": "PINNfor3agents.pt",
        "model_name": "CommNet",
        "landmarks": [13, 14, 0],
    },
    "PINNfor5agents" : {
        "checkpoint": "PINNfor5agents.pt",
        "model_name": "CommNet",
        "landmarks": [13, 14, 0, 1, 2],
    },
    "PINNfor8agents": {
        "checkpoint": "PINNfor8agents.pt",
        "model_name": "CommNet",
        "landmarks": [13, 14, 0, 1, 2, 3, 4, 5],
    },
    "PINNfor3D8agents": {
        "checkpoint": "PINNfor3d8agents.pt",
        "model_name": "Network3d",
        "landmarks": [13, 14, 0, 1, 2, 3, 4, 5],
    },
}

AGENT_COLORS = plt.cm.tab10.colors  # up to 10 distinct colors


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

def run_inference(nifti_path: str, checkpoint_path: str, model_name: str, landmarks: list):
    """Runs `DQN.py --task play` on a single volume for a given checkpoint /
    model_name / landmark set and returns:
      - agent_results: list of dicts, one per agent, each with
        {"agent": i, "landmark": landmark_id, "x": int, "y": int, "z": int}
      - trajectories: dict mapping agent index -> list of (step, x, y, z)
    """
    n_agents = len(landmarks)

    with tempfile.TemporaryDirectory() as tmp:
        files_list = os.path.join(tmp, "image_files.txt")
        with open(files_list, "w") as f:
            f.write(nifti_path + "\n")

        cmd = [
            sys.executable, "DQN.py",
            "--task", "play",
            "--load", checkpoint_path,
            "--files", files_list,
            "--file_type", "brain",
            "--landmarks", *[str(l) for l in landmarks],
            "--model_name", model_name,
            "--viz", "0",
        ]
        result = subprocess.run(
            cmd, cwd=SRC_DIR, capture_output=True, text=True, timeout=900
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Inference failed:\n" + result.stdout[-2000:] + "\n" + result.stderr[-2000:]
            )

        row = None
        trajectories = {i: [] for i in range(n_agents)}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("STEP_LOC:"):
                parts = line.split()
                try:
                    step = int(parts[1])
                    agent_idx = int(parts[2])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    if agent_idx in trajectories:
                        trajectories[agent_idx].append((step, x, y, z))
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

        # Each agent occupies an 8-field block after the leading run number:
        # [filename_i, x_i, y_i, z_i, landmark_x_i, landmark_y_i, landmark_z_i, distError_i]
        agent_results = []
        for i in range(n_agents):
            base = 1 + i * 8
            try:
                x = int(round(row[base + 1]))
                y = int(round(row[base + 2]))
                z = int(round(row[base + 3]))
            except (TypeError, ValueError, IndexError) as e:
                raise RuntimeError(
                    f"Could not parse agent {i} position from output row: {row}"
                ) from e
            agent_results.append({
                "agent": i,
                "landmark": landmarks[i],
                "x": x, "y": y, "z": z,
            })

        return agent_results, trajectories


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


def make_preview(nifti_path: str, agent_results: list):
    """One row of (Axial, Coronal, Sagittal) slices per agent, each centered
    on that agent's own detected landmark position."""
    image = sitk.ReadImage(nifti_path)
    array = sitk.GetArrayFromImage(image)
    zmax, ymax, xmax = array.shape

    n = len(agent_results)
    fig, axs = plt.subplots(n, 3, figsize=(9, 3.2 * n))
    if n == 1:
        axs = np.array([axs])

    for row_idx, res in enumerate(agent_results):
        x = min(max(res["x"], 0), xmax - 1)
        y = min(max(res["y"], 0), ymax - 1)
        z = min(max(res["z"], 0), zmax - 1)
        color = AGENT_COLORS[row_idx % len(AGENT_COLORS)]
        views = [
            (array[z, :, :], (x, y), "Axial"),
            (array[:, y, :], (x, z), "Coronal"),
            (array[:, :, x], (y, z), "Sagittal"),
        ]
        for ax, (slice_, point, title) in zip(axs[row_idx], views):
            ax.imshow(slice_, cmap="gray")
            ax.scatter([point[0]], [point[1]], c=[color], s=40, marker="+")
            ax.set_title(f"Agent {res['agent']} · LM {res['landmark']} · {title}")
            ax.axis("off")
    fig.tight_layout()
    return fig


def make_trajectory_gif(nifti_path: str, trajectory, out_path: str, fps: int = 8, color="#3a7bff"):
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
        ax.plot(x, y, "o", color=color, ms=7, mec="w", mew=0.6)
        ax.set_title(f"Step {step}", color="#ffcc00", fontsize=10)
        fig.suptitle("Agent search path", color="white", fontsize=11)

    anim = FuncAnimation(fig, render, frames=frames)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return out_path


def gif_as_html(gif_path, width=280):
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
    "Upload either a **NIfTI** (`.nii` / `.nii.gz`) or a **DICOM** (`.dcm`) file, "
    "then pick which trained agent configuration should analyze it."
)

col_upload = st.container()

with col_upload:
    st.subheader("⚙️ Model")
    model_choice = st.selectbox(
        "Agent configuration",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
        help="Single-agent models find one landmark. Multi-agent (CommNet/Network3d) "
             "models run several agents at once, each finding its own landmark.",
    )
    selected_config = MODEL_CONFIGS[model_choice]
    st.caption(
        f"Checkpoint: `{selected_config['checkpoint']}` · "
        f"model_name: `{selected_config['model_name']}` · "
        f"landmarks: {selected_config['landmarks']}"
    )

    st.subheader("📤 Upload")
    uploaded_files = st.file_uploader(
        "Upload MRI Images",
        accept_multiple_files=True,
        help="You can upload one or many NIfTI/DICOM images."
    )
    run_button = st.button("🚀 Start AI Analysis", type="primary", use_container_width=True)

# Image processing pipeline
if run_button:

    if not uploaded_files:
        st.error("Upload one or more MRI files first.")
        st.stop()

    checkpoint_path = os.path.join(MODELS_DIR, selected_config["checkpoint"])
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    model_name = selected_config["model_name"]
    landmarks = selected_config["landmarks"]
    n_agents = len(landmarks)

    results = []
    progress = st.progress(0)
    status = st.empty()
    top1, top2, top3 = st.columns(3)
    top1.metric("Uploaded", len(uploaded_files))
    top2.metric("Completed", 0)
    top3.metric("Failed", 0)

    with st.spinner("Running AI analysis..."):

        for idx, uploaded_file in enumerate(uploaded_files):

            status.info(
                f"Analyzing ({idx+1}/{len(uploaded_files)}) with {n_agents} agent(s): {uploaded_file.name}"
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

                agent_results, trajectories = run_inference(
                    image_path, checkpoint_path, model_name, landmarks
                )

                fig = make_preview(image_path, agent_results)

                gif_html = {}
                for i, res in enumerate(agent_results):
                    gif_fd, gif_path = tempfile.mkstemp(suffix=f".agent{i}.gif")
                    os.close(gif_fd)
                    color = AGENT_COLORS[i % len(AGENT_COLORS)]
                    color_hex = "#{:02x}{:02x}{:02x}".format(
                        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                    )
                    saved = make_trajectory_gif(
                        image_path, trajectories.get(i, []), gif_path, color=color_hex
                    )
                    if saved:
                        # Read the gif into an inline HTML <img> tag now, while
                        # the file still exists, then delete it immediately.
                        # (It would otherwise be gone by the time this result
                        # is rendered further down.)
                        gif_html[i] = gif_as_html(saved)
                        try:
                            os.remove(saved)
                        except OSError:
                            pass

                try:
                    os.remove(tmp.name)
                except OSError:
                    pass

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

                max_steps = max(
                    (trajectories[i][-1][0] for i in trajectories if trajectories[i]),
                    default="N/A",
                )

                report_lines = [f"Filename : {uploaded_file.name}", ""]
                for res in agent_results:
                    report_lines.append(
                        f"Agent {res['agent']} — Landmark {res['landmark']}: "
                        f"X={res['x']} Y={res['y']} Z={res['z']}"
                    )
                report_lines.append("")
                report_lines.append(f"Steps : {max_steps}")
                report_text = "\n".join(report_lines)

                # Save result (flattened per-agent columns for the CSV export)
                result_row = {"Filename": uploaded_file.name, "Agents": n_agents, "Steps": max_steps}
                for res in agent_results:
                    i = res["agent"]
                    result_row[f"Landmark_{i}"] = res["landmark"]
                    result_row[f"X_{i}"] = res["x"]
                    result_row[f"Y_{i}"] = res["y"]
                    result_row[f"Z_{i}"] = res["z"]
                results.append(result_row)

                progress.progress((idx + 1) / len(uploaded_files))

                # Display each result
                with st.expander(f"📁 {uploaded_file.name}", expanded=False):

                    st.pyplot(fig)

                    if gif_html:
                        st.markdown("**Agent search paths**")
                        gif_cols = st.columns(min(len(gif_html), 4) or 1)
                        for i, html in gif_html.items():
                            with gif_cols[i % len(gif_cols)]:
                                st.caption(f"Agent {i} (landmark {landmarks[i]})")
                                st.markdown(html, unsafe_allow_html=True)

                    st.text(report_text)

                    st.code(info)

            except Exception as e:

                results.append({
                    "Filename": uploaded_file.name,
                    "Agents": n_agents,
                    "Error": str(e),
                })

                with st.expander(
                    f"❌ {uploaded_file.name}"
                ):
                    st.exception(e)

    # ---------------- Summary ----------------

    df = pd.DataFrame(results)
    successful = sum(
        1 for r in results
        if "Error" not in r
    )

    failed = len(results) - successful

    top2.metric("Completed", successful)
    top3.metric("Failed", failed)

    st.success(
        f"Finished analysing {len(results)} image(s) with {n_agents} agent(s)."
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
