---
title: RL Medical Landmark Detection
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# RL-Medical: Anatomical Landmark Detection

Gradio demo wrapping a pretrained single-agent DQN (from the C-MARL /
RL-Medical project, physics-informed-reward-shaping adaptation) that locates
anatomical landmarks in brain MRI volumes.

Upload a `.nii` / `.nii.gz` brain scan and the agent will navigate to
landmark #13 (anterior commissure), reporting the voxel coordinates and
three orthogonal preview slices.

Only inference ("play" task) runs in this Space — training requires a GPU
and much longer sessions than a Space is meant for; see the original repo's
`README.md` / `README_ADAPTATION.md` in `src/` for training instructions.
