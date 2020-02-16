#ifndef ROBOT_HPP
#define ROBOT_HPP

#include "pid.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "person_kalman.hpp"
//#include <gazebo_ros/conversions/model_states.hpp>//geometry_msgs.hpp> //gazebo_msgs/msgs/model_states.hpp>
#include "gazebo_msgs/msg/model_states.hpp"


class Robot : public rclcpp::Node
{
  private:
    PersonKalman *person_kalman_;
    std::shared_ptr<rclcpp::Node> nh_;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr states_sub_;

  public:
    void statesCb(gazebo_msgs::msg::ModelStates::SharedPtr msg);
    Robot();
    ~Robot();
};


#endif
