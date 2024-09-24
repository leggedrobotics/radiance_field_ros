"""
sensor.py

Helper class for managing cameras and maintains their 
"""

from typing import Union
import torch
import numpy as np
import cv2
import time
import rospy
import ros_numpy
import scipy.spatial.transform as transform

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf.transformations import euler_matrix

from ros_nerf.data.ros_dataset import ROSDataset
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import CameraType


def tf_to_transform(tf_data: TransformStamped) -> torch.Tensor:
    """
    Convert a ROS TransformStamped message to NerfStudio compatible transformation matrix.

    Args:
        tf_data (TransformStamped): The ROS TransformStamped message.

    Returns:
        torch.Tensor: The PyTorch transformation matrix.

    """
    transform = ros_numpy.numpify(tf_data.transform)

    out = transform @ np.array(euler_matrix(np.pi/2, 0, -np.pi/2))
    T = torch.from_numpy(out[:3, :4])
    T = T.to(dtype=torch.float32)
    return T


def check_type_match(topic_name: str, cam_type: CameraType) -> bool:
    """
    Check if we need to use distortion parameters.

    Args:
        topic_name (str): The name of the topic.
        cam_type (CameraType): The camera type.

    Returns:
        bool: Whether to use distortion parameters.

    """
    if 'rect' in topic_name:
        CONSOLE.print(
            f"Warning: {topic_name} is a rectified image, but the camera type is {cam_type.name}, setting distortion to None")
        return False
    return True


def type_to_ns(ros_type: str) -> CameraType:
    """
    Convert ROS camera distortion model to NerfStudio camera type.

    Args:
        ros_type (str): The ROS camera distortion model.

    Returns:
        CameraType: The NerfStudio camera type.

    """

    if ros_type == "plumb_bob":
        return CameraType.PERSPECTIVE
    elif ros_type == "equidistant":
        return CameraType.FISHEYE
    else:
        raise ValueError(f"Unknown camera type: {ros_type}")


class Sensor():
    """
    Sensor class to manage camera information, ROS topics and callbacks.

    Args:
        name (str): The name of the sensor.
        image_topic (str): The image topic to subscribe to.
        camera_frame (str): The camera frame.
        base_frame (str): The base frame.
        info_topic (str, optional): The camera info topic. Defaults to None.
        depth_topic (str, optional): The depth topic. Defaults to None.
        use_preset_D (bool, optional): Whether to use the preset distortion matrix. Defaults to False.
        lookup (dict, optional): A dictionary of known poses to lookup. Defaults to None.
        blur_threshold (int, optional): The threshold for blur detection. Defaults to 80.
        hz (int, optional): The frequency to save images at. Defaults to 1.
        update_cbs (list, optional): List of update callbacks. Defaults to [].
    """

    def __init__(self, name, image_topic, camera_frame, base_frame, info_topic=None, depth_topic=None, use_preset_D=False, lookup=None, blur_threshold=80, hz=1, update_cbs=[], **kwargs):
        self.updated = False
        self.use_preset_D = use_preset_D
        self.base_frame = base_frame
        self.name = name
        self.img_topic = image_topic
        self.depth_topic = depth_topic
        self.info_topic = info_topic
        try:
            self.last_update_t = rospy.get_rostime().to_sec()
        except rospy.exceptions.ROSInitException as e:
            rospy.loginfo("Node not started, assuming eval mode...")
            self.last_update_t = time.time()
    
        self.blur_threshold = blur_threshold
        self.update_period = 1/hz
        self.frame = camera_frame
        if self.frame.startswith("/"):
            self.frame = self.frame[1:]
        self.H = None
        self.W = None
        self.dataset = None
        self.last_pose = None
        self.update_cbs = update_cbs

        self.warned = False
        self.count = 0

        self.lookup = lookup

        rospy.logdebug(
            f"Created sensor {self.name} with{'' if self.use_preset_D else 'out'} preset D, with{'out' if self.lookup is None else ''} lookup")

    def eval_check(self) -> bool:
        """
        Check if a particular image should be sent to the evaluation dataset.
        """
        
        return (self.dataset.split == 'val' and self.count % self.dataset.validation_factor == 0) or (self.dataset.split == 'train' and self.count % self.dataset.validation_factor != 0)

    def setup(self, dataset: ROSDataset):
        """
        Setup the sensor object and register callbacks.

        Args:
            dataset (ROSDataset): The ROSDataset object, or the saver object.

        """

        self.dataset = dataset

        rgb_type = CompressedImage if self.img_topic.endswith(
            "compressed") else Image

        rgb_sub = Subscriber(self.img_topic, rgb_type)

        if self.depth_topic is not None:
            # TODO: add support for compressed depth?
            depth_type = Image
            depth_sub = Subscriber(self.depth_topic, depth_type)
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=10, slop=0.1)
            sync.registerCallback(self.synced_cb)
            rospy.loginfo(
                f"Subscribed to {self.img_topic} and {self.depth_topic} for {self.name}")
        else:
            rgb_sub.registerCallback(self.image_cb)
            rospy.loginfo(f"Subscribed to {self.img_topic} for {self.name}")

        if self.info_topic is not None:
            rospy.Subscriber(self.info_topic, CameraInfo, self.info_cb)
        else:
            self.use_preset_D = True
            self.info_cb()

    def info_cb(self, info: CameraInfo = None):
        """
        Callback for camera info topic.

        Args:
            info (CameraInfo, optional): The camera info message. Defaults to None.
        """
        if self.updated:
            return

        if self.use_preset_D:
            rospy.loginfo(f"{self.name} using preset params")
            self.H = self.dataset.cameras[0].height
            self.W = self.dataset.cameras[0].width
            self.fx = self.dataset.cameras[0].fx
            self.fy = self.dataset.cameras[0].fy
            self.cx = self.dataset.cameras[0].cx
            self.cy = self.dataset.cameras[0].cy
            self.cam_type = self.dataset.cameras[0].camera_type
            self.D = self.dataset.cameras[0].distortion_params

        else:
            self.H = info.height
            self.W = info.width
            self.fx = info.K[0]
            self.fy = info.K[4]
            self.cx = info.K[2]
            self.cy = info.K[5]

            self.cam_type = type_to_ns(info.distortion_model).value


            if self.cam_type == CameraType.FISHEYE.value:
                self.D = torch.Tensor(
                    [info.D[0], info.D[1], info.D[2], info.D[3], 0, 0])

            elif self.cam_type == CameraType.PERSPECTIVE.value:
                self.D = torch.Tensor(
                    [info.D[0], info.D[1], info.D[4], 0, info.D[2], info.D[3]])


            # if we are dealing with a rectified image, we don't need distortion parameters
            if not check_type_match(self.info_topic, self.cam_type):
                self.D = torch.Tensor([0, 0, 0, 0, 0, 0])

        rospy.logdebug(
            f"{self.name} got info {self.H}x{self.W}, camera: {self.cam_type}, D: {self.D}, fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}")
        rospy.loginfo(f"{self.name} updated info")

        self.updated = True

    def synced_cb(self, image: Union[Image, CompressedImage], depth: Union[Image, CompressedImage]):
        """
        Callback for synchronized image and depth topics.
        """

        if self.save_checker():
            return  

        if self.save_img(image, image.header):
            self.save_depth(depth)

    def image_cb(self, image: Union[Image, CompressedImage]):
        """
        Callback for image topic.

        Args:
            image (Union[Image, CompressedImage]): The image message.
        """

        if self.save_checker():
            return
        
        rospy.loginfo(f"Got image {image.header.seq} for {self.dataset.split} dataset, {self.dataset.current_idx} / {self.dataset.num_images}")

        # self.dataset.iter_count += 1
        self.save_img(image, image.header)

    def save_depth(self, depth: Union[Image, CompressedImage]):
        """ 
        Save the depth image to the dataset.

        Args:
            depth (Union[Image, CompressedImage]): The depth image message.      
        """
        im_tensor = torch.frombuffer(depth.data, dtype=torch.float16).reshape(
            self.H, self.W, 1
        ).to(dtype=torch.float32)

        # -1 because we increment the index after saving the rgb image
        idx = self.dataset.current_idx - 1

        # resize to dataset size
        self.dataset.depth_tensor[idx] = torch.nn.functional.interpolate(im_tensor.permute(
            2, 0, 1).unsqueeze(0), size=self.dataset.resolution).squeeze(0).permute(1, 2, 0)

    def save_img(self, image: Union[Image, CompressedImage], header: Image.header):
        """
        Save the image to the dataset.

        Args:
            image (Union[Image, CompressedImage]): The image message.
            header (Image.header): The image header.
        """

        img = None
        if type(image) == CompressedImage:
            img = cv2.imdecode(np.frombuffer(
                image.data, np.uint8), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2:  # debayered image
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
            elif len(img.shape) == 3:
                img = img[:, :, ::-1].copy()
        else:
            img = ros_numpy.numpify(image)

        blur = cv2.Laplacian(img, cv2.CV_64F).var()

        if blur < self.blur_threshold:
            return False

        self.count += 1

        if not self.eval_check():
            return False

        im_tensor = torch.frombuffer(img, dtype=torch.uint8).reshape(
            self.H, self.W, -1
        )

        rospy.loginfo(
            f"{self.name} got image {header.seq} for {self.dataset.split} dataset, {self.dataset.current_idx} / {self.dataset.num_images}")

        im_tensor = im_tensor.to(dtype=torch.float32) / 255.0
        c2w = None
        if self.lookup is not None:
            rospy.loginfo(
                f"{header.seq} is {'in' if header.seq in self.lookup else 'not in'} lookup")
            if header.seq in self.lookup:
                out = np.array(self.lookup[header.seq])
                R1 = transform.Rotation.from_euler(
                    "y", 90, degrees=True).as_matrix()
                R2 = transform.Rotation.from_euler(
                    "x", 90, degrees=True).as_matrix()
                R = R1 @ R2
                out = R @ out
                out[:, -1] = out[:, -1] / 7
                c2w = torch.from_numpy(out).to(dtype=torch.float32)
            else:
                return False
        idx = self.dataset.current_idx
        # resize to dataset size
        self.dataset.image_tensor[idx] = torch.nn.functional.interpolate(im_tensor.permute(
            2, 0, 1).unsqueeze(0), size=self.dataset.resolution).squeeze(0).permute(1, 2, 0)
        y_scale = self.dataset.resolution[0] / self.H
        x_scale = self.dataset.resolution[1] / self.W

        if c2w is None:
            try:
                tf_data = self.dataset.buffer.lookup_transform(
                    self.base_frame, self.frame, header.stamp, rospy.Duration(1))
            except Exception as e:
                rospy.logerr(
                    f"Error getting transform: {e}, {self.frame} to {self.base_frame}")
                return False
            c2w = tf_to_transform(tf_data)

        device = self.dataset.cameras.device
        c2w = c2w.to(device)

        self.dataset.update_camera(idx, 
                                   camera_to_worlds=c2w, 
                                   times=header.stamp.to_sec(), 
                                   camera_type=self.cam_type, 
                                   fx=self.fx * x_scale, 
                                   fy=self.fy * y_scale, 
                                   cx=self.cx * x_scale, 
                                   cy=self.cy * y_scale, 
                                   width=self.W * x_scale, 
                                   height=self.H * y_scale, 
                                   distortion_params=self.D)
        
        # Update the map of ros image indices to nerfstudio image indices
        self.dataset.ros_to_nerf[header.seq] = idx

        self.dataset.updated_indices.append(idx)

        self.dataset.updated = True

        self.dataset.current_idx += 1
        return True

    def save_to_file_synced(self, rgb: Union[Image, CompressedImage], depth: Union[Image, CompressedImage]):
        """
        Save the synchronized image and depth topics to file.

        Args:
            rgb (Union[Image, CompressedImage]): The rgb image message.
            depth (Union[Image, CompressedImage]): The depth image message.
        """

        if self.save_checker():
            return

        # process rgb image
        rgb_image = None
        if type(rgb) == CompressedImage:
            rgb_image = cv2.imdecode(np.frombuffer(
                rgb.data, np.uint8), cv2.IMREAD_UNCHANGED)
            if len(rgb_image.shape) == 2:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BAYER_BG2RGB)
            elif len(rgb_image.shape) == 3:
                rgb_image = rgb_image[:, :, ::-1].copy()

        else:
            rgb_image = ros_numpy.numpify(rgb)

        # process depth image
        depth_image = None
        # TODO: check if this would ever happen or if it should be compressed depth
        if type(depth) == CompressedImage:
            depth_image = cv2.imdecode(np.frombuffer(
                depth.data, np.uint8), cv2.IMREAD_UNCHANGED)
            if len(depth_image.shape) == 2:
                depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BAYER_BG2RGB)
            elif len(depth_image.shape) == 3:
                depth_image = depth_image[:, :, ::-1].copy()

        else:
            depth_image = ros_numpy.numpify(depth)


        # bgr to rgb
        rgb_image = rgb_image[:, :, ::-1].copy()

        blur = cv2.Laplacian(rgb_image, cv2.CV_64F).var()

        if blur < self.blur_threshold:
            return

        rospy.logdebug(f"{self.name} got image")

        c2w = None

        # Check if we are looking up images from known poses
        if self.lookup is not None:
            if rgb.header.seq in self.lookup:
                out = np.array(self.lookup[rgb.header.seq])
                R1 = transform.Rotation.from_euler(
                    "y", 90, degrees=True).as_matrix()
                R2 = transform.Rotation.from_euler(
                    "x", 90, degrees=True).as_matrix()
                R = R1 @ R2
                out = R @ out
                out[:, -1] = out[:, -1] / 7
                c2w = torch.from_numpy(out).to(dtype=torch.float32)
            else:
                return

        # Save image to output folder
        cv2.imwrite(
            f"{self.dataset.output_path}/rgb/{self.name}_{rgb.header.seq:05}.png", rgb_image)
        cv2.imwrite(
            f"{self.dataset.output_path}/depth/{self.name}_{depth.header.seq:05}.tiff", depth_image)

        if c2w is None:
            try:
                tf_data = self.dataset.buffer.lookup_transform(
                    self.base_frame, self.frame, rgb.header.stamp, rospy.Duration(1))
                c2w = tf_to_transform(tf_data)

                c2w = torch.cat(
                    (c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0)
                c2w = c2w.tolist()

            except Exception as e:
                rospy.logerr(e)
                return

        d = self.D.cpu().numpy()

        self.dataset.current_idx += 1

        # Append per frame path, transform matrix and camera parameters
        if self.dataset.multi_cam:
            self.dataset.transforms['frames'].append({
                'file_path': f"./rgb/{self.name}_{rgb.header.seq:05}.png",
                'depth_file_path': f"./depth/{self.name}_{depth.header.seq:05}.png",
                'transform_matrix': c2w,
                'fl_x': str(self.fx),
                'fl_y': str(self.fy),
                'cx': str(self.cx),
                'cy': str(self.cy),
                'w': str(self.W),
                'h': str(self.H),
                'k1': str(d[0]),
                'k2': str(d[1]),
                'p1': str(d[2]),
                'p2': str(d[3]),
            })

        else:
            # Append per frame path and transform matrix
            self.dataset.transforms['frames'].append({
                'file_path': f"./rgb/{self.name}_{rgb.header.seq:05}.png",
                'depth_file_path': f"./depth/{self.name}_{depth.header.seq:05}.png",
                'transform_matrix': c2w
            })

            # Set camera parameters
            self.dataset.transforms['w'] = str(self.W)
            self.dataset.transforms['h'] = str(self.H)
            self.dataset.transforms['fl_x'] = str(self.fx)
            self.dataset.transforms['fl_y'] = str(self.fy)
            self.dataset.transforms['cx'] = str(self.cx)
            self.dataset.transforms['cy'] = str(self.cy)
            self.dataset.transforms['k1'] = str(d[0])
            self.dataset.transforms['k2'] = str(d[1])
            self.dataset.transforms['p1'] = str(d[2])
            self.dataset.transforms['p2'] = str(d[3])

    def save_img_to_file(self, msg: Union[Image, CompressedImage]):
        """
        Save the image to the output folder.

        Args:
            msg (Union[Image, CompressedImage]): The image message.
        """

        if self.save_checker():
            return

        img = None
        if type(msg) == CompressedImage:
            img = cv2.imdecode(np.frombuffer(
                msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2:  # debayered image
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
            elif len(img.shape) == 3:
                img = img[:, :, ::-1].copy()
        else:
            img = ros_numpy.numpify(msg)

        # bgr to rgb
        img = img[:, :, ::-1].copy()

        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if blur < self.blur_threshold:
            return

        rospy.logdebug(f"{self.name} got image")

        c2w = None

        # Check if we are looking up images from known poses
        if self.lookup is not None:
            if msg.header.seq in self.lookup:
                out = np.array(self.lookup[msg.header.seq])
                R1 = transform.Rotation.from_euler(
                    "y", 90, degrees=True).as_matrix()
                R2 = transform.Rotation.from_euler(
                    "x", 90, degrees=True).as_matrix()
                R = R1 @ R2
                out = R @ out
                out[:, -1] = out[:, -1] / 7
                c2w = torch.from_numpy(out).to(dtype=torch.float32)
            else:
                return

        # Save image to output folder
        cv2.imwrite(
            f"{self.dataset.output_path}/rgb/{self.name}_{msg.header.seq:05}.png", img)

        if c2w is None:
            try:
                tf_data = self.dataset.buffer.lookup_transform(
                    self.base_frame, self.frame, msg.header.stamp, rospy.Duration(1))
                c2w = tf_to_transform(tf_data)

                # TODO: swap this to numpy/agnostic
                c2w = torch.cat(
                    (c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0)
                c2w = c2w.tolist()

            except Exception as e:
                rospy.logerr(e)
                return

        d = self.D.cpu().numpy()

        self.dataset.current_idx += 1

        if self.dataset.multi_cam:
            # Append per frame path, transform matrix and camera parameters
            self.dataset.transforms['frames'].append({
                'file_path': f"./rgb/{self.name}_{msg.header.seq:05}.png",
                'transform_matrix': c2w,
                'fl_x': str(self.fx),
                'fl_y': str(self.fy),
                'cx': str(self.cx),
                'cy': str(self.cy),
                'w': str(self.W),
                'h': str(self.H),
                'k1': str(d[0]),
                'k2': str(d[1]),
                'p1': str(d[2]),
                'p2': str(d[3]),
            })

        else:
            # Append per frame path and transform matrix
            self.dataset.transforms['frames'].append({
                'file_path': f"./rgb/{self.name}_{msg.header.seq:05}.png",
                'transform_matrix': c2w
            })

            # Set camera parameters
            self.dataset.transforms['w'] = str(self.W)
            self.dataset.transforms['h'] = str(self.H)
            self.dataset.transforms['fl_x'] = str(self.fx)
            self.dataset.transforms['fl_y'] = str(self.fy)
            self.dataset.transforms['cx'] = str(self.cx)
            self.dataset.transforms['cy'] = str(self.cy)
            self.dataset.transforms['k1'] = str(d[0])
            self.dataset.transforms['k2'] = str(d[1])
            self.dataset.transforms['p1'] = str(d[2])
            self.dataset.transforms['p2'] = str(d[3])

    def save_checker(self) -> bool:
        """ Used in the image saving methods to check if it should continue or not """

        # If we haven't updated the camera parameters yet, don't save images
        if not self.updated:
            return True

        # If the service call says to stop, don't save images
        if self.dataset.run == False:
            return True

        # If there hasnt been enough time since the last update, don't save images
        now = rospy.get_rostime().to_sec()
        if now - self.last_update_t < self.update_period:
            return True

        # If we have reached the max number of images, don't save images
        if self.dataset.current_idx >= self.dataset.max_imgs:
            if not self.warned:
                self.warned = True
                rospy.logwarn(
                    f"Reached max images {self.dataset.max_imgs} on {self.name}")
            return True

        # Save the image and update last updated time
        self.last_update_t = now

        return False
