#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

// https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

int main(int argc, char** argv) {
  std::filesystem::path path = std::filesystem::absolute(argv[0]).parent_path();
  const cv::Mat image = cv::imread(path.string() + "/../image.png");

  std::vector<cv::Mat> planes;
  cv::split(image, planes);

  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  cv::threshold(gray_image, gray_image, 0, UINT8_MAX,
                cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

  cv::imshow("", gray_image);
  cv::waitKey(0);

  // // noise removal
  // cv::Mat opening;
  // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  // cv::morphologyEx(gray_image, opening, cv::MORPH_OPEN, kernel,
  //                  cv::Point(-1, -1), 3);
  // cv::Mat blur;
  // cv::GaussianBlur(blur, opening, cv::Size(5, 5), 0);
  // cv::threshold(blur, blur, 1, UINT8_MAX, cv::THRESH_BINARY);
  // kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  // cv::morphologyEx(opening, opening, cv::MORPH_CLOSE, kernel, cv::Point(-1,
  // -1),
  //                  3);

  // // sure background area
  // cv::Mat sure_bg;
  // cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 2);

  // // Finding sure foreground area
  // cv::Mat dist_transform;
  // cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
  // cv::distanceTransform(sure_bg, dist_transform, cv::DIST_L2, 5);
  return EXIT_SUCCESS;
}
