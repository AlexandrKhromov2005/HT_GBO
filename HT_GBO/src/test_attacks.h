#ifndef TEST_ATTACKS_H
#define TEST_ATTACKS_H

#include <opencv2/opencv.hpp>

#include "test_attacks.h"

// A1: Сжатие JPEG с качеством 40
cv::Mat jpegCompression40(const cv::Mat& img);

// A2: Сжатие JPEG с качеством 70
cv::Mat jpegCompression70(const cv::Mat& img);

// A5: Медианный фильтр 3x3
cv::Mat medianFilter3x3(const cv::Mat& img);

// A6: Медианный фильтр 5x5
cv::Mat medianFilter5x5(const cv::Mat& img);

// A7: Гауссовский фильтр 3x3
cv::Mat gaussianFilter3x3(const cv::Mat& img);

// A8: Гауссовский фильтр 5x5
cv::Mat gaussianFilter5x5(const cv::Mat& img);

// A9: Шум 0.2%
cv::Mat saltPepperNoise02(const cv::Mat& img);

// A10: Шум 1%
cv::Mat saltPepperNoise1(const cv::Mat& img);

// AA: Поворот на 15° и обратно
cv::Mat rotate15(const cv::Mat& img);

// AB: Поворот на 30° и обратно
cv::Mat rotate30(const cv::Mat& img);

// AC: Увеличение 0.5x и обратно
cv::Mat scale05(const cv::Mat& img);

// AD: Увеличение 4x и обратно
cv::Mat scale4(const cv::Mat& img);

// AE: Сдвиг (10,10) и обратно
cv::Mat translate10(const cv::Mat& img);
// AF: Сдвиг (20,40) и обратно
cv::Mat translate20_40(const cv::Mat& img);

#endif