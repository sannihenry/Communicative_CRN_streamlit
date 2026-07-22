"""
RL-Medical / physics-informed landmark detector.

Loads the bundled single-agent DQN checkpoint (trained to find landmark #13 -
the anterior commissure - in brain MRI) and runs it in inference-only
("play") mode on a NIfTI volume the user uploads.

This wraps the existing CLI (`src/DQN.py --task play`) as a subprocess so the
original, unmodified training/inference code is reused as-is.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
import spaces

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
MODEL_PATH = os.path.join(SRC_DIR, "data", "models", "BrainMRI", "SingleAgent.pt")
LANDMARK_ID = 13  # what SingleAgent.pt was trained on


@spaces.GPU
def run_inference(nifti_path: str):
    """Runs `DQN.py --task play` on a single volume and returns the row
    printed by the logger with the agent's final (x, y, z) voxel position."""

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

        # The logger prints each results row as a Python list literal.
        # Find the last one that starts with an episode index (an int).
        row = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                try:
                    parsed = ast.literal_eval(line)
                    if isinstance(parsed, list) and isinstance(parsed[0], int):
                        row = parsed
                except (ValueError, SyntaxError):
                    continue
        if row is None:
            raise RuntimeError("Could not parse model output:\n" + result.stdout[-2000:])

        # row = [episode, filename, agent_x, agent_y, agent_z, landmark_x/N/A, ...]
        x, y, z = row[2], row[3], row[4]
        return int(round(x)), int(round(y)), int(round(z))


def make_preview(nifti_path: str, x: int, y: int, z: int):
    """Renders the axial/coronal/sagittal slices through the predicted point."""
    image = sitk.ReadImage(nifti_path)
    array = sitk.GetArrayFromImage(image)  # z, y, x
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


def detect(file):
    if file is None:
        raise gr.Error("Upload a .nii or .nii.gz brain MRI volume first.")
    x, y, z = run_inference(file.name)
    fig = make_preview(file.name, x, y, z)
    text = (
        f"Predicted landmark {LANDMARK_ID} (anterior commissure) voxel position: "
        f"x={x}, y={y}, z={z}"
    )
    return fig, text


with gr.Blocks(title="RL-Medical Landmark Detector") as demo:
    gr.Markdown(
        """
        # Anatomical Landmark Detection (C-MARL / RL-Medical)
        Upload a brain MRI in NIfTI format (`.nii` or `.nii.gz`). The bundled
        single-agent DQN checkpoint will navigate to landmark **#13**
        (anterior commissure) and report the voxel coordinates it converges on.

        """
    )
    with gr.Row():
        inp = gr.File(label="Brain MRI (.nii / .nii.gz)", file_types=[".nii", ".gz"])
    btn = gr.Button("Detect landmark", variant="primary")
    out_plot = gr.Plot(label="Predicted location")
    out_text = gr.Textbox(label="Result")

    btn.click(fn=detect, inputs=inp, outputs=[out_plot, out_text])

if __name__ == "__main__":
    demo.launch()
