#ifndef OF_H
#define OF_H

#include "attacks.h"
#include "metrics.h"
#include <opencv2/opencv.hpp>
#include <utility>
#include <functional>

using WeightAttackPair = std::pair<double, std::function<cv::Mat()>>;
inline double objectiveFunction(const std::vector<cv::Mat&> originalImages, std::vector<cv::Mat&> watermarkedImages, const std::vector<cv::Mat&> originalWMs, const std::vector<cv::Mat&> extractedWMs);


#endif // !OF_H
