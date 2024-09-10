#!/usr/bin/env python3

"""
ros_dataparser.py

Replaces Colmap or other dataparsers to directly use and manage ROS data online.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import rospy
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

from ros_nerf.utils.sensor import Sensor
from ros_nerf.utils.ros_config import ROSConfig

from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ROSDataParserConfig(DataParserConfig):
    """ROS config file parser config."""

    _target: Type = field(default_factory=lambda: ROSDataParser)
    """target class to instantiate"""

    config: ROSConfig = ROSConfig()
    """ROS parameters config object."""

    scale_factor: float = 0.1
    """How much to scale the camera origins by."""


@dataclass
class ROSDataParser(DataParser):
    """ROS DataParser"""

    config: ROSDataParserConfig
    includes_time = True

    def __init__(self, config: ROSDataParserConfig):
        super().__init__(config=config)
        self.ros: ROSConfig = config.config
        self.scale_factor: float = config.scale_factor

    def get_dataparser_outputs(self, split="train"):
        dataparser_outputs = self._generate_dataparser_outputs(split)
  
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train"):
        """
        Generate the dataparser outputs for the ROS dataset.

        """
        
        num_images = self.ros.num_images   
        if split == "val":
            num_images = num_images // self.ros.validation_factor

        CONSOLE.log(f"Loading {num_images} images for {split} dataparser")

        image_height = self.ros.height
        image_width = self.ros.width
        fx = self.ros.fx
        fy = self.ros.fy
        cx = self.ros.cx
        cy = self.ros.cy
        
        self.distort = torch.tensor(self.ros.distort, dtype=torch.float32)

        camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[
            :, :-1, :
        ]

        # broadcast height and width to match num_images
        self.image_height = torch.tensor(num_images * [image_height], dtype=torch.int)
        self.image_width = torch.tensor(num_images * [image_width], dtype=torch.int)
        self.fx = torch.tensor(num_images * [fx], dtype=torch.float32)
        self.fy = torch.tensor(num_images * [fy], dtype=torch.float32)
        self.cx = torch.tensor(num_images * [cx], dtype=torch.float32)
        self.cy = torch.tensor(num_images * [cy], dtype=torch.float32)


        # TODO: Make the arrays jagged when this is supported by torch/nerfstudio

        # in x,y,z order
        scene_size = self.ros.aabb
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-scene_size, -scene_size, -scene_size],
                    [scene_size, scene_size, scene_size],
                ],
                dtype=torch.float32,
            )
        )

        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            height=self.image_height,
            width=self.image_width,
            times=torch.arange(num_images, dtype=torch.float32),
            distortion_params=self.distort,
            camera_type=CameraType.PERSPECTIVE,
        )

        sensors = []
        
        # if this isnt the training split, append the split to the name of the sensor
        if split != "train":
            for camera in self.ros.cameras:
                camera["name"] = f"{camera['name']}_{split}"


        for camera in self.ros.cameras:
            sensors.append(Sensor(**camera, **self.ros.__dict__))

        image_filenames = []
        metadata = {
            "split": split,
            "validation_factor": self.ros.validation_factor,
            "sensors": sensors,
            "num_images": num_images,
            "image_height": image_height,
            "image_width": image_width,
            "num_start": self.ros.num_start,
            "base_frame": self.ros.base_frame,
            "blur_threshold": self.ros.blur_threshold,
            "hz": self.ros.hz,
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
    
# Create the hook for linking the dataparser with Nerfstudio
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

ROSDataparser = DataParserSpecification(config=ROSDataParserConfig())