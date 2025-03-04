#ifndef OF_H
#define OF_H

#include "attacks.h"
#include "metrics.h"
#include <opencv2/opencv.hpp>
#include <utility>
#include <functional>

using WeightAttackPair = std::pair<double, std::function<cv::Mat()>>;


#endif // !OF_H
