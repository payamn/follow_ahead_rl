cmake_minimum_required(VERSION 3.5)
project(follow_ahead_rl)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_LIBS}
)

add_executable(robot
  src/pid.cpp
  src/person_kalman.cpp
  src/robot.cpp)
ament_target_dependencies(robot rclcpp std_msgs cv_bridge OpenCV)

install(TARGETS
  robot 
  DESTINATION lib/${PROJECT_NAME})

ament_package()

