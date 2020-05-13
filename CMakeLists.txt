cmake_minimum_required(VERSION 2.8.3)
project(follow_ahead_rl)

find_package(catkin REQUIRED COMPONENTS rospy)

# add_message_files(
#   FILES  # e.g. Floats.msg HeaderString.msg
# )
# add_service_files(
#   DIRECTORY srv
#   FILES AddTwoInts.srv BadTwoInts.srv
# )

## Generate services in the 'srv' folder
# add_service_files(
#  FILES  # e.g. Floats.srv HeaderString.srv
#)

## Generate added messages and services with any dependencies
# generate_messages()

catkin_package(
  CATKIN_DEPENDS
)
