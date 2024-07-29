cmake_minimum_required(VERSION 3.0.2)
project(nerf_teleoperation)


find_package(catkin REQUIRED COMPONENTS nerf_teleoperation_msgs nerf_view_controller)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS nerf_teleoperation_msgs nerf_view_controller
)