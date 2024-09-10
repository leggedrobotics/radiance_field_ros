#!/usr/bin/env python3

"""
ros_dataloader.py

Really simple dataloaders that skip the undistortion and tensor setup steps in the normal pipeline.
"""
import warnings
from typing import Union, List, Optional, Callable, Any, Sized, Tuple, Dict, Type

from nerfstudio.utils.rich_utils import CONSOLE
import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import EvalDataloader
from ros_nerf.data.ros_dataset import ROSDataset

import rospy

class ROSDataloader(DataLoader):

    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = None,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
            CONSOLE.print(
                "Excluding image from device",
                style="bold dodger_blue1",
            )
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device
        self.dataset = dataset
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.device = device

        self.collate_fn = collate_fn

        self.batch = {}


    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.dataset.data_dict.items():
            dev = self.device
            if k in self.exclude_batch_keys_from_device:
                dev = None
            if isinstance(v, torch.Tensor):
                batch[k] = v[0: self.dataset.current_idx-1, ...].to(device=dev)
            elif isinstance(v, list):
                batch[k] = torch.tensor(v[0: self.dataset.current_idx-1]).to(device=dev)
        return batch

    def __iter__(self):
        while True:
            if len(self.batch.keys()) == 0 or len(self.batch['image_idx']) < len(self.dataset):
                self.batch = self._get_updated_batch()

            batch = self.batch
            yield batch



class ROSEvalDataLoader(EvalDataloader):
    """ Based on the base random indices dataloader, but for streaming images 
        Args:
            input_dataset: Input dataset to sample from.
            data_loader: Dataloader to use current idx from
            device: Device to load data to.
    """
    def __init__(self, 
                 input_dataset: InputDataset, 
                 device: Union[torch.device, str] = "cpu",
                 idx: Optional[Tuple[int]] = None,
                 **kwargs):
        
        super().__init__(input_dataset, device, **kwargs)

        # use constant seed for reproducibility
        self.count = 0
        self.ros = False

        if idx is None:
            # if we dont have a list of indices, then we will generate ones based on the initial 
            self.idx = list(range(self.input_dataset.num_start))
        else:
            # Convert the ros sequence indices to the input dataset indices
            self.idx = idx
            self.ros = True

        # Init will duplicate the cameras and move them if the device is different, use one of the existing sets instead
        if device == self.input_dataset.cameras.device:
            self.cameras = self.input_dataset.cameras
        else:
            self.cameras = self.input_dataset.ray_cameras

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        # randomly choose images that are less than current_idx
        if self.count < len(self.idx) and self.count < self.input_dataset.current_idx:
            idx = self.idx[self.count]
            if self.ros:
                try:
                    idx = self.input_dataset.ros_to_nerf[idx]
                except:
                    rospy.logerr(f"Index {idx} not found in saved images!")
                    return None
            camera, batch = self.get_camera(idx)
            self.count += 1
            return camera, batch
        raise StopIteration

        # ray_bundle, batch = self.get_data_from_image_idx(image_idx)
        # return ray_bundle, batch
    