#ifndef ATTACKS_H
#define ATTACKS_H

#include <opencv2/opencv.hpp>

cv::Mat applyJPEGCompression(const cv::Mat& image, int quality);

cv::Mat applyMedianFilter(const cv::Mat& image, int kernelSize);

cv::Mat applyGaussianFilter(const cv::Mat& image, int kernelSize, double sigma);

cv::Mat applySaltPepperNoise(const cv::Mat& image, double noiseRatio);

cv::Mat applyRotationAttack(const cv::Mat& image, double angle);


cv::Mat applyScalingAttack(const cv::Mat& image, double scaleFactor);

cv::Mat applyTranslationAttack(const cv::Mat& image, int offsetX, int offsetY);

#endif // ATTACKS_H
