# UWB-Radar-Hand-Pose-Estimation

This repo contains code for manuscript: <span style="color:blue"><b>Radar-Based Hand Pose Estimation: Advancing Human-Computer Interaction with FMCW Radar on Edge Devices</b></span>.

## 🎥 Demo
### 🌙 Demo Results in Dark Scenes
| (a) The poses estimated from FMCW Radar signals | (b) The poses estimated from Camera signals |
|-------------------|--------------------|
| ![](doc/images/radar_dark_cases_demo.jpg) | ![](doc/images/cam_dark_cases_demo.jpg) |

*Visual comparison between hand key points estimated by proposed RadarPose model and [MediaPipe Hands](https://arxiv.org/abs/2006.10214) on dark scenes.*
<p align="center">
  <em>
    Visual comparison between hand key points estimated by proposed RadarNet model and 
    <a href="https://arxiv.org/abs/2006.10214">MediaPipe Hands</a> 
    on dark scenes.
  </em>
</p>

### 🔅 Demo Results in Light Scenes
<p align="center">
  <img src="doc/images/Radar-Cam-Pose-1.png" width="48.5%">
  <img src="doc/images/Radar-Cam-Pose-2.png" width="49%">
</p>

<p align="center">
  <em>
    Visual comparison between hand key points estimated by 
    <a href="https://arxiv.org/abs/2006.10214">MediaPipe Hands</a> 
    and proposed RadarNet model. Blue dots are key points estimated by the visual model while red ones are from RadarNet.
  </em>
</p>

## 🚀 Quickstart
```bash
git clone https://github.com/thetuantrinh/UWB-Radar-Hand-Pose-Estimation.git
```

### 🛠 Environment
The original project was developed on python 3.9.0. We encourage you to create the same python version for reproduce purposes by creating python3.9 with conda by the following script:
```bash
conda create --name HPE python==3.9
conda activate HPE
```
***Then install all required libraries:***
```bash
pip3 install -r requirements.txt
```

### 📚 Training

⚠️ **Important:** Please update the default dataset directory in `scripts/train_hpc.sh` to the absolute path of your dataset.

### 1. Structure the project
Before running training scripts, first structure the project by executing:
```bash
bash scripts/structure_project.sh
```

### 📊 Tracking Training with Weights & Biases (Wandb)

[Wandb](https://wandb.ai) is a great tool for experiment tracking and visualization.  

1. Install with pip:
```bash
pip install wandb
``` 

#### ⚠️ Potential Problems (HPC without Internet)

If you're training on a machine without internet connection (e.g., an HPC compute node), Wandb will not work online.  
To fix this, run in **offline mode**:
```bash
wandb offline
```
After training, sync all locally saved Wandb logs to the cloud:
```bash
wandb sync your-local-wandb-log-folder/offline-run*
```
👉 Remeber to replace `your-local-wandb-log-folder` with the path to your actual Wandb logs directory.

