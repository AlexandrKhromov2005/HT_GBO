#ifndef TEST_ATTACKS_H
#define TEST_ATTACKS_H

#include <opencv2/opencv.hpp>

#include "test_attacks.h"

// A1: ������ JPEG � ��������� 40
cv::Mat jpegCompression40(const cv::Mat& img);

// A2: ������ JPEG � ��������� 70
cv::Mat jpegCompression70(const cv::Mat& img);

// A5: ��������� ������ 3x3
cv::Mat medianFilter3x3(const cv::Mat& img);

// A6: ��������� ������ 5x5
cv::Mat medianFilter5x5(const cv::Mat& img);

// A7: ����������� ������ 3x3
cv::Mat gaussianFilter3x3(const cv::Mat& img);

// A8: ����������� ������ 5x5
cv::Mat gaussianFilter5x5(const cv::Mat& img);

// A9: ��� 0.2%
cv::Mat saltPepperNoise02(const cv::Mat& img);

// A10: ��� 1%
cv::Mat saltPepperNoise1(const cv::Mat& img);

// AA: ������� �� 15� � �������
cv::Mat rotate15(const cv::Mat& img);

// AB: ������� �� 30� � �������
cv::Mat rotate30(const cv::Mat& img);

// AC: ���������� 0.5x � �������
cv::Mat scale05(const cv::Mat& img);

// AD: ���������� 4x � �������
cv::Mat scale4(const cv::Mat& img);

// AE: ����� (10,10) � �������
cv::Mat translate10(const cv::Mat& img);
// AF: ����� (20,40) � �������
cv::Mat translate20_40(const cv::Mat& img);

#endif