from typing import Union

import torch
from typing_extensions import Literal

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
import rospy
import tf2_ros
from pathlib import Path
import os
import json
import cv2
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import track


class ROSDataset(InputDataset):
    """
    Overrides the normal dataset with support for depth images (if in the config) and standard images.
    Mainly acts as a placeholder while the images actually get inserted into the dataset with the callbacks.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
    ):
        rospy.loginfo("ROS Dataset init")
        super().__init__(dataparser_outputs, scale_factor)

        assert (
            "num_images" in dataparser_outputs.metadata.keys()
            and "sensors" in dataparser_outputs.metadata.keys()
            and "base_frame" in dataparser_outputs.metadata.keys()
            and len(dataparser_outputs.metadata["sensors"]) > 0
        )

        self.use_depth = any([cam.depth_topic for cam in self.metadata["sensors"]])
        
        # self.cameras = self.cameras.to("cuda:0")
        self.ray_cameras = None

        if self.use_depth:
            rospy.loginfo("Depth model used")

        self.sensors = self.metadata["sensors"]
        self.base_frame = self.metadata["base_frame"]
        self.hz = self.metadata["hz"]
        self.split = self.metadata["split"]

        self.current_idx = 0
        self.updated = False
        self.run = True

        self.num_images = self.metadata["num_images"]
        self.num_start = self.metadata["num_start"]
        self.max_imgs = self.num_images


        assert self.num_images > 0
        
        self.image_height = self.metadata["image_height"]
        self.image_width = self.metadata["image_width"]
        self.resolution = (self.image_height, self.image_width)

        self.split = self.metadata["split"]
        self.validation_factor = self.metadata["validation_factor"]

        self.buffer = tf2_ros.Buffer(rospy.Duration(120.0))
        self.listener = tf2_ros.TransformListener(self.buffer)

        # Setup and subscribe each camera in the sensor list
        for sensor in self.sensors:
            sensor.setup(self)


        # empty list with num_images elements
        self.image_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
        )

        if self.use_depth:
            self.depth_tensor = torch.ones(
                self.num_images, self.image_height, self.image_width, 1, dtype=torch.float32
            )
        else:
            self.depth_tensor = None

        self.image_indices = torch.arange(self.num_images)
        self.times = torch.zeros(self.num_images)

        self.updated_indices = []

        self.data_dict = {
            "image": self.image_tensor,
            "image_idx": self.image_indices,
        }

        self.ros_to_nerf = {}

        if self.use_depth:
            self.data_dict["depth_image"] = self.depth_tensor

    def update_camera(self, camera_idx: int, **kwargs):
        """
        This function updates the camera pose and intrinsics in both the ray cameras (gpu) and cpu cameras.
        TODO: This is really ugly and shouldn't be necessary. However the cameras are copied over in the VanillaDataManager
        https://github.com/nerfstudio-project/nerfstudio/blob/3a90cb529f893fbf89625a915a53a7a71b97a575/nerfstudio/data/datamanagers/base_datamanager.py#L499
        So we need to update both the GPU version and CPU version of the cameras.
        """

        assert self.ray_cameras is not None, "Ray cameras not initialized"
        for key, value in kwargs.items():
            if hasattr(self.cameras, key):
                getattr(self.cameras, key)[camera_idx] = value
                getattr(self.ray_cameras, key)[camera_idx] = value
            else:
                rospy.logwarn(f"Key {key} not found in camera dict")

    def get_indices(self):
        """ Overrides the functions in traditional datamanagers which randomly sample indices and instead randomly samples up to current_idx """
        indices = torch.arange(self.current_idx)
        shuffled_indices = torch.randperm(len(indices))
        indices = list(indices[shuffled_indices])
        for i in indices:
            camera = self.cameras[i]
            img = self[i]

        return indices

    def reset_cameras(self):
        self.cameras.height = torch.tensor(self.num_images * [[self.image_height]])
        self.cameras.width = torch.tensor(self.num_images * [[self.image_width]])

    def __len__(self):
        # Return the number of images, or the number that will be loaded before training
        return self.num_images
    
    def __deepcopy__(self, memo):
        """
        Fakes the copy by returning a list of all current images and their data dicts
        Only used for full datamanger's fixed indices
        """
        data = []
        for idx in range(self.current_idx):
            data.append(self.get_data(idx))

        return data

    def save(self, path: Path):
        """
        Saves the dataset to a file, helpful for debugging.
        """

        # Start by writing all current cameras to a transform.json
        cams = {"camera_model": "OPENCV", "frames": []}
        # make the rgb directory
        rgb_dir = path / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        if self.use_depth:
            depth_dir = path / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)

        for i in track(range(self.current_idx), description="Saving images..."):
            cams['frames'].append({
                "file_path": f"rgb/{i:05}.png",
                "transform_matrix": self.cameras.camera_to_worlds[i].tolist()
            })
            if self.use_depth:
                cams['frames'][-1]['depth_file_path'] = f"depth/{i:05}.png"
                # save depth image
                depth_image = self.depth_tensor[i].squeeze().cpu().numpy()
                cv2.imwrite(str(depth_dir / f"{i:05}.png"), depth_image)
            
            # save rgb image
            rgb_image = self.image_tensor[i].squeeze().cpu().numpy()
            cv2.imwrite(str(rgb_dir / f"{i:05}.png"), rgb_image)

        with open(path / "transforms.json", "w") as f:
            json.dump(cams, f, indent=4)
        
    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> dict:
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = {"image_idx": image_idx, "image": self.image_tensor[image_idx]}

        if self.use_depth:
            data['depth_image'] = self.depth_tensor[image_idx]

        return data
