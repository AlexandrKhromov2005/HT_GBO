#ifndef PROCESSWM_H
#define PROCESSWM_H

#include "POB.h"
#include "affine.h"
#include <array>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

using KEY_A = std::array<int, 6>;
using KEY_B = std::vector<int>;

cv::Mat process(cv::Mat wm);
cv::Mat restore(cv::Mat wm);

#endif // !PROCESSWM_H
