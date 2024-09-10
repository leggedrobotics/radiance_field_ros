# Radiance Fields for Robotic Teleoperation

[Project Page](https://leggedrobotics.github.io/rffr.github.io/)

## Introduction

This package combines NerfStudio with ROS to provide a multi-camera system for data collection and visualization. This includes a ROS node for NerfStudio training, helper scripts to interface with the NerfStudio CLI, as well as an RVIZ plugin for visualization.

## Installation

Start by cloning this repository into your catkin workspace and using [rosdep](http://wiki.ros.org/rosdep) to install system packages. 

Next you need to link this package to NerfStudio. If NerfStudio is already installed and in a conda environment, activate that first then go into this folder and run `pip install -e .`.

If NerfStudio is not already installed, you can install it with the following commands:
```bash

conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
pip install --upgrade pip

conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# This may take a while to build
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Finally install the package and NerfStudio with
pip install -e .

```


Next build the package as normal with `catkin build nerf_teleoperation`, then source the workspace. Double check installation worked with `ns-ros -h` this should display the help message along with all the installed NerfStudio methods. Also verify that the RViz plugin works by launching RViz with the workspace sourced and switching to the NerfViewController type in the Views panel.


## Usage
### Configuration

Both the online and offline modes are setup using the configuration files located in the config/ folder. Their parameters and layout are as follows:
```yaml
---
# Base config for the NerfStudio ROS sensors

height: int  # image height
width: int   # image width
fx: float    # focal length in pixels [optional]
fy: float    # focal length in pixels [optional]
cx: float    # principal point in pixels [optional]
cy: float    # principal point in pixels [optional]
k1: float    # radial distortion coefficient [optional]
k2: float    # radial distortion coefficient [optional]
k3: float    # radial distortion coefficient [optional]
p1: float    # tangential distortion coefficient [optional]
p2: float    # tangential distortion coefficient [optional]
base_frame: string   # root frame for the camera system, should be fixed such as "world" or "map"
use_preset_D: bool   # whether to force the preset distortion parameters [optional]
num_images: int      # number of images to capture
num_start: int       # number of images to capture before training starts, defaults to all images [optional]
hz: int              # capture rate in hz
blur_threshold: int  # threshold for blur detection between 0 and 100
publish_hz: int      # rate to publish the a splat pointcloud2 at [optional]
cameras:             # array of cameras to subscribe too
	- name: string         # name of the camera
		image_topic: string  # topic for the rgb image
		depth_topic: string  # topic for the depth image [optional]
		info_topic:  string  # topic for the camera info
		camera_frame: string # frame of the camera

```

### Offline Training

```bash
ns-ros-save --config_path <config_file> --save_path <output_folder> --no-run-on-start [OPTIONS]
```

To just capture data in the NerfStudio format for later use with a non-ROS NerfStudio install, run the `ns-ros-save` command. This will start the ROS node, and save the data into an rgb/, and depth/ folder if applicable. Various `[OPTIONS]` arguments may also be specified to overwrite any of the components in the config file shown above. For example `--base-frame odom` or `--num_start 5` to overwrite the parameters from the config. The program will automatically save the data once the preset number of images is reached, or the rosservice is trigged with `rosservice call /save_transforms`, or the program is terminated with `ctrl-c`. The capturing can be paused and resumed using the `rosservice call /toggle`.


### Online Training
````bash
ns-ros <model-name> --config_path <config_file> [OPTIONS]
````

To train a model with the ROS node, run the `ns-ros` command with some `<model-name>`. Any model installed and working with `ns-train` should work with the ROS node. The `<config_file>` specifies the path to sensor config detailing all the ROS topics to subscribe to and node parameters. Various `[OPTIONS]` arguments may also be specified to overwrite any of the components in the config file shown above. For example `--base-frame odom` or `--num_start 5` to overwrite the parameters from the config. This will start the ROS node, and wait for data to start streaming. It will then begin training the model based on the config file and allow visualization from the NerfStudio viewer, or interaction via the action server. The data capture can be paused and resumed using the `rosservice call /set_capture <boolsrv>`, and for splat reconstruction, the `rosservice call /save_splat` will save the splat.ply file to the output folder.


### Visualization
With the workspace sourced, the RVIZ plugin should be linked, allowing you to select "NerfViewController" from the Type dropdown in the Views panel. This will send render requests to the NerfStudio node, and display the renders in the RVIZ window.

For more immersive rendering, checkout the Unity VR project at [nerf-teleoperation-unity](https://github.com/leggedrobotics/unity_ros_teleoperation).

### Datasets
Some sample ROS bag datasets can be found [here](https://drive.google.com/drive/folders/1_Z-Z5WJOUWyvzGk0lrY_ORTcB7Wi2xQw?usp=sharing) along with informatiopn about which config file goes with each bag.

## Troubleshooting
If there is an issue with the conda environment and the catkin_ws, you can make catkin use the conda environment for python path by cleaning and rebuilding the workspace with the conda environment activated.

## Citing
If you use this project in your work please cite [this paper](https://arxiv.org/abs/2407.20194):
```text
@article{wildersmith2024rfteleoperation,
  author    = {Maximum Wilder-Smith, Vaishakh Patil, Marco Hutter},
  title     = {Radiance Fields for Robotic Teleoperation},
  journal   = {arXiv},
  year      = {2024},
}
```
