#ifndef OF_H
#define OF_H

#include "attacks.h"
#include "metrics.h"
#include <opencv2/opencv.hpp>
#include <utility>
#include <functional>

using WeightAttackPair = std::pair<double, std::function<cv::Mat()>>;
double objectiveFunction(std::vector<cv::Mat> originalImages, std::vector<cv::Mat> watermarkedImages, std::vector<cv::Mat> originalWMs, std::vector<cv::Mat> extractedWMs);


#endif // !OF_H
