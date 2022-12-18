#include <iostream>

#include <librealsense2/rs.hpp>  // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>    // Include OpenCV API

int main() {
  rs2::config config;
  config.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_BGR8);
  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;
  // Start streaming with default recommended configuration
  pipe.start(config);

  for (int i = 0; i < 100; ++i) {
    pipe.wait_for_frames();
  }
  rs2::frameset data =
      pipe.wait_for_frames();  // Wait for next set of frames from the camera
  rs2::video_frame color_frame = data.get_color_frame();
  cv::Mat image(cv::Size(color_frame.get_width(), color_frame.get_height()),
                CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
  std::vector<cv::Mat> planes;

  cv::imwrite("image.png", image);
  // cv::split(image, planes);
  cv::imshow("image", image);
  // cv::imshow("image r", planes[0]);
  // cv::imshow("image g", planes[1]);
  // cv::imshow("image b", planes[2]);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
