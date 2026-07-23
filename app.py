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
import pydicom

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


def prepare_image(file):
    """
    Accepts:
      • .nii
      • .nii.gz
      • .dcm

    Returns a NIfTI filename because the RL model expects NIfTI input.
    """

    filename = file.name.lower()

    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return file.name

    if filename.endswith(".dcm"):
        ds = pydicom.dcmread(file.name)

        image = sitk.GetImageFromArray(ds.pixel_array)

        tmp_dir = tempfile.mkdtemp()
        nifti_path = os.path.join(tmp_dir, "converted.nii.gz")

        sitk.WriteImage(image, nifti_path)

        return nifti_path

    raise gr.Error("Unsupported file format.")
def detect(file):

    if file is None:
        raise gr.Error("Please upload a NIfTI or DICOM image.")

    nifti_path = prepare_image(file)

    x, y, z = run_inference(nifti_path)

    fig = make_preview(nifti_path, x, y, z)

    text = f"""
✅ Analysis Complete

Detected Landmark : #{LANDMARK_ID}

Voxel Coordinates

X : {x}

Y : {y}

Z : {z}
"""

    return fig, text

css = """
.gradio-container{
    max-width:1500px !important;
}

h1{
    text-align:center;
}

.section{
    border-radius:15px;
    padding:15px;
    background:#f8fafc;
    border:1px solid #e5e7eb;
}

.footer{
    text-align:center;
    font-size:13px;
    color:gray;
}
"""

with gr.Blocks(
    css=css,
    title="Communicative RL Medical Imaging"
) as demo:

    gr.Markdown(
        """
# 🧠 Communicative Reinforcement Learning Medical Imaging Platform

### Physics-Informed Landmark Detection for Brain MRI

Upload either

- ✅ NIfTI (.nii / .nii.gz)

or

- ✅ DICOM (.dcm)

The AI automatically detects anatomical landmarks.
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("## 📤 Upload")

            file_input = gr.File(
                label="Medical Image",
                file_types=[
                    ".nii",
                    ".nii.gz",
                    ".dcm"
                ]
            )

            detect_btn = gr.Button(
                "🚀 Start AI Analysis",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.ClearButton(
                components=[file_input],
                value="🗑 Clear"
            )

            gr.Markdown("---")

            metadata = gr.Textbox(
                label="📋 Image Information",
                lines=10,
                interactive=False
            )

        with gr.Column(scale=2):

            with gr.Tabs():

                with gr.Tab("🖼 Visualization"):

                    output_image = gr.Plot(
                        label="Landmark Detection"
                    )

                with gr.Tab("📈 Analysis"):

                    report = gr.Textbox(
                        label="AI Report",
                        lines=18,
                        interactive=False
                    )

                with gr.Tab("📄 Download"):

                    report_file = gr.File(
                        label="Download Generated Report"
                    )

    gr.Markdown("---")

    with gr.Accordion("⚙ Technical Details", open=False):

        gr.Markdown(
            """
### Model

Communicative Deep Reinforcement Learning

### Framework

PyTorch

### Input

NIfTI / DICOM

### Output

3D Anatomical Landmark Coordinates

### Institution

Research Prototype
"""
        )

    gr.Markdown(
        """
<div class='footer'>
Communicative Reinforcement Learning Landmark Detection • Powered by Gradio
</div>
"""
    )

    def analyse(file):

        fig, text = detect(file)

        try:

            image_path = prepare_image(file)

            if image_path.endswith(".nii") or image_path.endswith(".nii.gz"):

                img = nib.load(image_path)

                arr = img.get_fdata()

                info = f"""
Filename : {os.path.basename(image_path)}

Shape : {arr.shape}

Data Type : {arr.dtype}

Dimensions : {len(arr.shape)}

Minimum : {arr.min():.2f}

Maximum : {arr.max():.2f}
"""

            else:

                ds = pydicom.dcmread(image_path)

                info = f"""
Patient : {getattr(ds,'PatientName','Unknown')}

Modality : {getattr(ds,'Modality','Unknown')}

Rows : {ds.Rows}

Columns : {ds.Columns}

Manufacturer :
{getattr(ds,'Manufacturer','Unknown')}
"""

        except:

            info = "Unable to extract metadata."

        report_path = "analysis_report.txt"

        with open(report_path,"w") as f:

            f.write(text)

        return (
            fig,
            text,
            info,
            report_path
        )

    detect_btn.click(
        analyse,
        inputs=file_input,
        outputs=[
            output_image,
            report,
            metadata,
            report_file
        ]
    )

demo.launch()
