# Minimum cmake version
cmake_minimum_required(VERSION 2.8.3)

# Create the project
project(nerf_teleoperation_msgs)

# 
find_package(catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
    geometry_msgs
    sensor_msgs
    actionlib_msgs
)

add_action_files(
    DIRECTORY action
    FILES
    NerfRenderRequest.action
)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    actionlib_msgs
)

catkin_package(
    CATKIN_DEPENDS 
    geometry_msgs
    sensor_msgs
)
