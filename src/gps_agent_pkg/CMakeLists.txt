cmake_minimum_required(VERSION 2.4.6)

project(gps_agent_pkg)


find_package(catkin
  REQUIRED COMPONENTS
    std_msgs
    sensor_msgs
    controller_interface
    pluginlib
    roscpp
    roslib
    rospy
    control_toolbox
    realtime_tools
    geometry_msgs
    tf
    kdl_parser
    cmake_modules
    genmsg
)

find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)


# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)
# TODO: is this the correct way to set build types?


add_message_files(
FILES
  CaffeParams.msg
  DataRequest.msg
  LinGaussParams.msg
  RelaxCommand.msg
  TfActionCommand.msg
  TfParams.msg
  ControllerParams.msg
  DataType.msg
  PositionCommand.msg
  SampleResult.msg
  TfObsData.msg
  TrialCommand.msg
)
generate_messages()


catkin_package(
INCLUDE_DIRS
  include
  ${EIGEN3_INCLUDE_DIRS}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
) 

#Include boost
find_package(Boost REQUIRED COMPONENTS thread)
include_directories(${Boost_INCLUDE_DIRS}) 


set(DDP_FILES
  src/robotplugin.cpp
  src/armplugin.cpp
  src/jacoarmplugin.cpp
  src/sample.cpp
  src/sensor.cpp
  src/controller.cpp
  src/lingausscontroller.cpp
  src/positioncontroller.cpp
  src/trialcontroller.cpp
  src/tfcontroller.cpp
  src/encodersensor.cpp
  src/encoderfilter.cpp
  src/rostopicsensor.cpp
  src/util.cpp
)


add_library(gps_agent_lib ${DDP_FILES})
target_link_libraries(gps_agent_lib ${catkin_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(gps_agent_lib gps_agent_pkg_generate_messages_cpp)
