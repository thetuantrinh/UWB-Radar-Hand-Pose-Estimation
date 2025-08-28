# UWB-Radar-Hand-Pose-Estimation

This repo contains code for manuscript: 
<h2 style="color:blue;">
Radar-Based Hand Pose Estimation: Advancing Human-Computer Interaction with FMCW Radar on Edge Devices
</h2>


| (a) The poses estimated from FMCW Radar signals | (b) The poses estimated from Camera signals |
|-------------------|--------------------|
| ![](doc/images/radar_dark_cases_demo.jpg) | ![](doc/images/cam_dark_cases_demo.jpg) |

*Visual comparison between hand key points estimated by proposed RadarPose model and [MediaPipe Hands](https://arxiv.org/abs/2006.10214) on dark scenes.*



```bash
git clone https://github.com/thetuantrinh/UWB-Radar-Hand-Pose-Estimation.git
```

## Environment
The original project was developed on python 3.9.0. We encourage you to create the same python version for reproduce purposes by creating python3.9 with conda by the following script:
```bash
conda create --name HPE python==3.9
conda activate HPE
```
***Then install all required libraries:***
```bash
pip3 install -r requirements.txt
```
### Traning
Please remember to change your default data dir in the train_hpc.sh script by the absolute path to your dataset.
***Before running scripts for training, run the following scripts to structure the project:***
```bash
bash scripts/structure_project.sh
```
To train the model with different settings, you can see parameters being passed to the train.py function in scripts/train_hpc.sh, or simply run it with:
```bash
sbatch scripts/train_hpc.sh
```
## Tracking training with Wandb
Wandb is a great tool for visualization. Install it with pip, then register for an account on wandb.ai to get an API for use. Follow the page on how to achieve this. 

### Potential problems
If you're training on a machine that doesn't have internet connection, let's say a compute node on a HPC cluster, your code will not run since wandb doesn't have internet. To run it and then sync to wandb cloud for synchonization, open the terminal or pass the following line of code to your `sh` script:
```bash
wandb offline
```
To push all these locally saved wandb logs and information to the cloud (wandb.ai), simple run `wandb sync your-local-wandb-log-folder/offline-run*`. Remeber to replace `your-local-wandb-log-folder` by your own

