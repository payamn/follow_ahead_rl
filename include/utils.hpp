#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

cv::Mat theta2RotationMatrix(float theta);
float rotationMatrix2Theta(cv::Mat rotation_matrix);

cv::Mat xytheta2TransformationMatrix(cv::Mat xytheta);
cv::Mat xytheta2TransformationMatrix(cv::Point3f xytheta);
cv::Point3f transformPoint(tf::StampedTransform transform, cv::Point3f point);
cv::Point3f transformPoint(tf::Transform transform, cv::Point3f point);
float vectorAngle(cv::Point2f vector1, cv::Point2f vector2);
