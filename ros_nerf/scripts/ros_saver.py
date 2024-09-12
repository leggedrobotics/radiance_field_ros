#!/usr/bin/env python3
"""
ros_saver.py

This script runs a special node that saves the images and camera poses to data/ in conventional nerfstudio format.
This includes RGB/ and Depth/ (if applicable) folders for images, and a transforms.json file.
Running ns-train from this folder will train on the data using standard nerfstudio training.
"""

import json
import yaml
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections import namedtuple

import numpy as np
import rospy
import tf2_ros
import torch
import tyro
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nerfstudio.utils.rich_utils import CONSOLE
from rich import box
from rich.align import Align
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeRemainingColumn)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_srvs.srv import Empty, EmptyResponse
from typing_extensions import Annotated

from ros_nerf.utils.sensor import Sensor, type_to_ns, tf_to_transform
from ros_nerf.utils.ros_config import ROSConfig

@dataclass
class SaverConfig:
    """Configuration for the ROS saver node."""

    save_path: Path = Path.cwd() / "data"
    """Path to save the data to. Defaults to data/ in the current working directory."""

    ros: Annotated[ROSConfig, tyro.conf.arg(name="")] = ROSConfig()
    """ROS configuration settings"""


@dataclass
class ROSSaver:
    """Class to create a ROS node that saves images and poses for nerfstudio training."""

    def main(self, config: SaverConfig):
        """Creates the ROS node to save"""

        rospy.init_node("ros_saver", anonymous=True)
        signal.signal(signal.SIGINT, self.shutdown)

        logging.getLogger('rosout').handlers[0] = RichHandler(
                console=CONSOLE
                )
        
        rospy.loginfo("ROS Saver node started")

        rospy.Service("save_transforms", Empty, self.write_json)
        rospy.Service("toggle", Empty, self.toggle)

        # Transforms holds the data that will go into transforms.json
        # Currently only supports perspective camera model
        self.transforms = {
            "camera_model": "OPENCV",
            "frames": []
            }
        
        self.config = config.ros
        name = self.config.update()

        # Check if we are dealing with multiple cameras/intrinsic parameters or just one
        self.multi_cam = len(self.config.cameras) > 1

        if not self.multi_cam:
            # Try to populate the transforms dict with the config file, if multiple cameras only store this per frame
            self.transforms.update({
                "w": self.config.width,
                "h": self.config.height,
                "fl_x": getattr(self.config, "fx", 0),
                "fl_y": getattr(self.config, "fy", 0),
                "cx": getattr(self.config, "cx", 0),
                "cy": getattr(self.config, "cy", 0),
                "k1": getattr(self.config, "k1", 0),
                "k2": getattr(self.config, "k2", 0),
                "p1": getattr(self.config, "p1", 0),
                "p2": getattr(self.config, "p2", 0)
            })

        # for naming compatibility with normal ros mode
        self.cameras = [
            namedtuple('Camera', ['height', 'width', 'fx', 'fy', 'cx', 'cy', 'camera_type', 'distortion_params'])(
            **{
                'height': self.config.height,
                'width': self.config.width,
                'fx': getattr(self.config, "fx", 0),
                'fy': getattr(self.config, "fy", 0),
                'cx': getattr(self.config, "cx", 0),
                'cy': getattr(self.config, "cy", 0),
                'camera_type': type_to_ns('plumb_bob'),
                'distortion_params': torch.tensor([getattr(self.config, "k1", 0), getattr(self.config, "k2", 0), getattr(self.config, "p1", 0), getattr(self.config, "p2", 0), 0])
            })
        ]

        self.max_imgs = self.config.num_images
        self.run = self.config.run_on_start

        self.depth = any(['depth_topic' in sensor for sensor in self.config.cameras])

        self.rgb_start = -1
        if self.depth:
            rospy.loginfo("Capturing depth data")
            self.depth_start = -1

        self.current_idx = 0

        self.output_path = config.save_path / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.make_dirs()

        # Create cameras from config file
        self.sensors = [Sensor(**s, **self.config.__dict__) for s in self.config.cameras]

        self.buffer = tf2_ros.Buffer(rospy.Duration(120.0))
        tf2_ros.TransformListener(self.buffer)

        for sensor in self.sensors:
            sensor.dataset = self

            msg_type = Image if not sensor.img_topic.endswith("compressed") else CompressedImage

            rgb_sub = Subscriber(sensor.img_topic, msg_type)

            if sensor.depth_topic:
                depth_sub = Subscriber(sensor.depth_topic, Image)
                ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 1, 0.1)
                ts.registerCallback(sensor.save_to_file_synced)
                rospy.loginfo(f"Subscribed to {sensor.img_topic} and {sensor.depth_topic}")
            else:
                rgb_sub.registerCallback(sensor.save_img_to_file)
                rospy.loginfo(f"Subscribed to {sensor.img_topic} to RGB topic")
            
            if sensor.info_topic:
                rospy.Subscriber(sensor.info_topic, CameraInfo, sensor.info_cb)
            else:
                sensor.use_preset_D = True
                sensor.info_cb()
                rospy.loginfo(f"No camera info topic for {sensor.img_topic}, using default values")
                

        # Wait for some data to come in and update one of the sensors
        with CONSOLE.status("[bold green] Waiting for data..."):
            while not (rospy.is_shutdown() or any([sensor.updated for sensor in self.sensors])):
                time.sleep(0.1)

        # Display the progress of loading in new data
        with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TimeRemainingColumn(), MofNCompleteColumn(),TextColumn("Images loaded"), console=CONSOLE) as progress:
            capture_task = progress.add_task("Capturing data...", total=self.max_imgs)
            while not rospy.is_shutdown():
            
                progress.update(capture_task, completed=len(self.transforms["frames"]), description="Capturing data..." if self.run else "Paused")
                if len(self.transforms["frames"]) >= self.max_imgs:
                    rospy.loginfo("Reached max number of images")
                    self.shutdown()                    
                time.sleep(0.1)


    def shutdown(self, sig=None, frame=None):
        """Shutdown handler"""
        rospy.loginfo("Shutting down ROS Saver node...")
        self.write_json()

        # Display the command to train on the data using ns-train and nerfacto
        relative_path = Path(os.path.relpath(self.output_path, Path.cwd()))
        panel_content = Panel(Align(f"ns-train {'depth-' if self.depth else ''}nerfacto nerfstudio-data --orientation-method none --data {relative_path}", "center"), title="Training Command", box=box.ROUNDED, border_style="bold", padding=(1,2))
        CONSOLE.print(panel_content)

        sys.exit(0)


    def write_json(self, _ = None):
        """Writes the transforms to a json file"""
        rospy.loginfo(f"Writing transforms {' with' if self.depth else ' without'} depth")
        self.run = False

        with open(f'{self.output_path}/transforms.json','w') as f:
            out = self.transforms
            f.write(json.dumps(out,indent=4))
        
        rospy.loginfo(f"Saved to {self.output_path}")

        resp = EmptyResponse()
        return resp

    def toggle(self, _ = None):
        """Toggles the node on or off"""
        self.run = not self.run
        rospy.loginfo("Running..." if self.run else "Stopping...")
        resp = EmptyResponse()
        return resp

    def make_dirs(self):
        """Creates the output directories"""
    
        self.output_path.mkdir(parents=True, exist_ok=True)
    
        (self.output_path / "rgb").mkdir(parents=True, exist_ok=True)
    
        if self.depth:
            (self.output_path / "depth").mkdir(parents=True, exist_ok=True)

def entrypoint():
    """Entrypoint for tyro"""
    tyro.extras.set_accent_color("#367ac6")
    
    # TODO: use ros_runner sensor config
    ROSSaver().main(tyro.cli(SaverConfig))


if __name__ == "__main__":
    entrypoint()