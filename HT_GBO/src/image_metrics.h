#ifndef IMAGE_METRICS_H
#define IMAGE_METRICS_H

#include <opencv2/opencv.hpp>

// ���������� ������������������ ������ (MSE) ����� ����� �������������
double computeImageMSE(const cv::Mat& image1, const cv::Mat& image2);

// ���������� ��������� �������� ������� � ���� (PSNR) ����� ����� �������������
double computeImagePSNR(const cv::Mat& image1, const cv::Mat& image2);

// ���������� ������� ������ (BER) ��� �������� �����������
double computeImageBER(const cv::Mat& image1, const cv::Mat& image2);

// ���������� ������������� ���������� (NCC) ����� ����� �������������
double computeImageNCC(const cv::Mat& img1, const cv::Mat& img2);

double computeImageSSIM(const cv::Mat& img1, const cv::Mat& img2);


#endif // IMAGE_METRICS_H
