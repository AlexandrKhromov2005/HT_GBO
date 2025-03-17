#ifndef PROCESSWM_H
#define PROCESSWM_H

#include "POB.h"
#include "affine.h"
#include <array>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <windows.h>
#include <filesystem>
#include <iostream>
#include "config.h"

using KEY_A = std::array<int, 6>;
using KEY_B = std::vector<int>;

std::vector<std::pair<int, int>> process(cv::Mat wm);
cv::Mat restore(std::vector<std::pair<int, int>> pixels);
KEY_B get();
bool writeVectorToFile(const std::vector<int>& keys);

#endif // !PROCESSWM_H
