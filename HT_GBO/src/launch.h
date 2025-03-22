#ifndef LAUNCH_H
#define LAUNCH_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "test_attacks.h"
#include "image_metrics.h"
#include "GBO.h"
#include "image_processing.h"
#include <functional>
#include <fstream>
#include <filesystem>

using AttackFunction = std::function<cv::Mat(const cv::Mat&)>;
using MetricCalculator = std::function<double(const cv::Mat&, const cv::Mat&)>;

struct AttackConfig {
    std::string name;
    AttackFunction attack;
    bool use_cropped_comparison = false;
};

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm);
void embend_wm(const std::string& image, const std::string& new_image, const std::string& wm);
void get_wm(const std::string& image, const std::string& new_wm);
cv::Mat get_wm(const cv::Mat& cv_image);

#endif // LAUNCH_H