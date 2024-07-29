# Minimum cmake version
cmake_minimum_required(VERSION 3.0.2)

# Create the project
project(nerf_view_controller)

# Load RViz and check version compile options
find_package(catkin REQUIRED 
    COMPONENTS 
    jsk_rviz_plugins
    rviz
    nerf_teleoperation_msgs
)

set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

if(rviz_QT_VERSION VERSION_LESS "5")
    message(STATUS "Using Qt4 based on the rviz_QT_VERSION: ${rviz_QT_VERSION}")
    find_package(Qt4 ${rviz_QT_VERSION} EXACT REQUIRED QtCore QtGui)
    include(${QT_USE_FILE})
else()
    message(STATUS "Using Qt5 based on the rviz_QT_VERSION: ${rviz_QT_VERSION}")
    find_package(Qt5 ${rviz_QT_VERSION} EXACT REQUIRED Core Widgets)
    set(QT_LIBRARIES Qt5::Widgets)
endif()

include_directories(${QT_INCLUDE_DIR})
add_definitions(-DQT_NO_KEYWORDS)


set(${PROJECT_NAME}_SRCS
    src/${PROJECT_NAME}/nerf_cam.cpp
)

set(${PROJECT_NAME}_HDRS
    include/${PROJECT_NAME}/nerf_cam.hpp
)

set(${PROJECT_NAME}_INCLUDE_DIRECTORIES
    include
    ${CMAKE_CURRENT_BINARY_DIR}
)

catkin_package(
    INCLUDE_DIRS
    include
    LIBRARIES
    ${PROJECT_NAME}
    CATKIN_DEPENDS 
    rviz
    nerf_teleoperation_msgs
)

add_definitions("-D${PROJECT_NAME}_VERSION=\"${${PROJECT_NAME}_VERSION}\"")

include_directories(
    ${${PROJECT_NAME}_INCLUDE_DIRECTORIES}
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_library(
    ${PROJECT_NAME}
    ${${PROJECT_NAME}_SRCS}
    ${${PROJECT_NAME}_HDRS}
)


add_dependencies(
    ${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(
    ${PROJECT_NAME}
    ${QT_LIBRARIES}
    ${catkin_LIBRARIES}
)

install(
    FILES
    plugin_description.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
    TARGETS
    ${PROJECT_NAME}
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
    DIRECTORY
    include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)