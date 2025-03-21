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
cv::Mat restore(std::vector<std::pair<int, int>> pixels, unsigned char layer);
KEY_B get();
void update(int n);
KEY_A read();
bool writeVectorToFile(const std::vector<int>& keys);
std::filesystem::path get_exe_directory();

#endif // !PROCESSWM_H
