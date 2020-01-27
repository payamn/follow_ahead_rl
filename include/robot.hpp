#ifndef ROBOT_HPP
#define ROBOT_HPP

#include "pid.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "person_kalman.hpp"

class Robot : public rclcpp::Node
{
  private:
    PersonKalman *person_kalman_;
    std::shared_ptr<rclcpp::Node> nh_;

  public:
    Robot();
  
    ~Robot();
};


#endif  
