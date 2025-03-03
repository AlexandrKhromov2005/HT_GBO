#ifndef METRICS_H
#define METRICS_H

#include <opencv2/opencv.hpp>

double computePSNR(const cv::Mat& I1, const cv::Mat& I2);

double computeSSIM(const cv::Mat& I1, const cv::Mat& I2);

double computeNC(const cv::Mat& watermarkOriginal, const cv::Mat& watermarkExtracted);

double computeBER(const cv::Mat& binaryImage1, const cv::Mat& binaryImage2);

#endif // METRICS_H
