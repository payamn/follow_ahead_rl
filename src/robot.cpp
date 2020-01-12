#include "robot.hpp"
#include "config.h"

Robot::Robot(): Node("follow_classic")
{
  cv::Mat Q = cv::Mat::zeros(NUM_STATES, NUM_STATES, CV_32F);
  Q.at<float>(X_T_IDX, X_T_IDX) = X_T_PROCESS_NOISE_VAR;
  Q.at<float>(Y_T_IDX, Y_T_IDX) = Y_T_PROCESS_NOISE_VAR;
  Q.at<float>(X_T_1_IDX, X_T_1_IDX) = X_T_1_PROCESS_NOISE_VAR;
  Q.at<float>(Y_T_1_IDX, Y_T_1_IDX) = Y_T_1_PROCESS_NOISE_VAR;
  Q.at<float>(VEL_IDX, VEL_IDX) = VEL_PROCESS_NOISE_VAR;
  Q.at<float>(THETA_IDX, THETA_IDX) = THETA_PROCESS_NOISE_VAR;

  cv::Mat R = cv::Mat::zeros(2, 2, CV_32F);
  R.at<float>(0, 0) = X_T_MEASUREMENT_NOISE_VAR;
  R.at<float>(1, 1) = Y_T_MEASUREMENT_NOISE_VAR;

  cv::Mat P = cv::Mat::eye(NUM_STATES, NUM_STATES, CV_32F);
  P.at<float>(0, 0) = X_T_INIT_ERROR_VAR;
  P.at<float>(1, 1) = Y_T_INIT_ERROR_VAR;
  P.at<float>(2, 2) = X_T_1_INIT_ERROR_VAR;
  P.at<float>(3, 3) = Y_T_1_INIT_ERROR_VAR;
  P.at<float>(4, 4) = VEL_INIT_ERROR_VAR;
  P.at<float>(5, 5) = THETA_INIT_ERROR_VAR;

  // pub_waypoints_ = nh_.advertise<sensor_msgs::PointCloud>("/person_follower/waypoints", 1);

  person_kalman_ = new PersonKalman(0.1, Q, R, P);
  RCLCPP_INFO(this->get_logger(), "%s\n", "hi");

}

Robot::~Robot()
{
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Robot>());
  rclcpp::shutdown();
  return 0;
}
