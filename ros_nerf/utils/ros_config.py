#!/usr/bin/env python3
"""
ros_config.py

Helper file to manage the configuration of the ROS nodes, both saver and runner.
This creates and manages sensors as well as the execution parameters.
"""

from typing_extensions import Annotated
from pathlib import Path
from dataclasses import dataclass
import tyro
from tyro.conf import Suppress
import yaml
from rich.pretty import pprint
from nerfstudio.utils.rich_utils import CONSOLE
import warnings
from typing import Union, Tuple

# Supress warnings from tyro
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ROSConfig:
    """Configuration for the ROS node."""

    config_path: Path = Path(__file__).parent.parent / "config" / "arm.yaml"
    """Path to the config file for sensors. Defaults to the arm depth config file."""

    run_on_start: bool = True
    """Whether to start the node running on start. Defaults to True."""

    base_frame: str = "map"
    """Frame to use as the base frame for the sensor. Defaults to config settings."""

    use_preset_D: bool = False
    """Whether to use the preset distortion matrix from the config file. Defaults to False."""

    num_images: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 0
    """Number of images to save. 0 means use the config file. Defaults to config settings."""

    num_start: Annotated[int, tyro.conf.arg(aliases=["-s"])] = -1
    """Number of images to store before beginning training. Defaults to config settings."""

    hz: float = 0.0
    """Frequency to save images at. Defaults to config settings."""

    blur_threshold: float = 0.0
    """Threshold for blur detection, higher means less selective. Defaults to config settings."""

    validation_factor: float = 0.0
    """Every Nth image is used for validation. Defaults to config settings."""

    aabb: float = 1.0
    """Default AABB size for the scene box. Defaults to 1.0."""

    eval_indicies: Union[Tuple[int,...], Path, None] = None
    """Indicies of images to use for evaluation based on image header seq. Defaults to None."""

    height: Suppress[int] = 0
    """Height of the image, populated at runtime from the config."""

    width: Suppress[int] = 0
    """Width of the image, populated at runtime from the config."""

    fx: Suppress[float] = 500.0
    """Focal length in x, populated at runtime from the config, or 500.0 if not found."""

    fy: Suppress[float] = 500.0
    """Focal length in y, populated at runtime from the config, or 500.0 if not found."""

    cx: Suppress[float] = 0
    """Center of image in x, populated at runtime from the config and defaults to half the width."""

    cy: Suppress[float] = 0
    """Center of image in y, populated at runtime from the config and defaults to half the height."""

    distort: Suppress[list] = None
    """Distortion parameters, populated at runtime from the config."""

    cameras: Suppress[list] = None
    """List of sensors to use, populated at runtime."""

    publish_hz: Suppress[float] = 0.0
    """Frequency to publish the splat at, defaults to 0 which means no publishing."""


    def update(self) -> str:
        """
        Updates the config file with the current values.

        Returns:
            str: The name of the experiment to the config file.
        """

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        # Check everything is valid
        self.check_config(config)

        # Update the config object with values in the file
        for key, value in config.items():
            if getattr(ROSConfig, key) == getattr(self, key) or getattr(self, key) == None:
                self.__setattr__(key, value)

        # Initialize the principal point to the center of the image 
        if "cx" not in config:
            self.cx = self.width / 2
        if "cy" not in config:
            self.cy = self.height / 2
        
        # Initialize the distortion parameters to 0 if not specified
        k1 = config["k1"] if "k1" in config else 0.0
        k2 = config["k2"] if "k2" in config else 0.0
        k3 = config["k3"] if "k3" in config else 0.0
        k4 = config["k4"] if "k4" in config else 0.0
        p1 = config["p1"] if "p1" in config else 0.0
        p2 = config["p2"] if "p2" in config else 0.0

        self.distort = [k1, k2, k3, k4, p1, p2]

        # Convert the eval indicies to a tuple if they are a file
        if isinstance(self.eval_indicies, Path):
            with open(self.eval_indicies, "r") as f:
                self.eval_indicies = tuple([int(i) for i in f.readlines()])

        return self.config_path.stem


    def check_config(self, config: dict) -> None:
        """Checks the config file for required parameters."""

        if "width" not in config or "height" not in config:
            raise ValueError("Width and height must be specified in config file")
        
        if "base_frame" not in config:
            raise ValueError("Base frame must be specified in config file")
        
        if "num_images" not in config:
            raise ValueError("Number of images must be specified in config file")
        
        if "cameras" not in config or len(config["cameras"]) == 0:
            raise ValueError("Cameras must be specified in config file")
        
        if "hz" not in config:
            raise ValueError("Frequency must be specified in config file")
        
        if "blur_threshold" not in config:
            raise ValueError("Blur threshold must be specified in config file")
        
        if isinstance("eval_indicies", Path) and not self.eval_indicies.exists():
            raise FileNotFoundError(f"File {self.eval_indicies} does not exist")
        
    
    def print_to_terminal(self):
        CONSOLE.rule("ROS Config")
        pprint(self)