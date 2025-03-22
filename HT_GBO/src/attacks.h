#ifndef ATTACKS_H
#define ATTACKS_H

#include <opencv2/opencv.hpp>

// JPEG Compression
cv::Mat applyJPEGCompression40(const cv::Mat& image);
cv::Mat applyJPEGCompression70(const cv::Mat& image);

// JPEG2000 Compression
cv::Mat applyJPEG2000Compression41(const cv::Mat& image);
cv::Mat applyJPEG2000Compression71(const cv::Mat& image);

// Median Filter
cv::Mat applyMedianFilter3x3(const cv::Mat& image);
cv::Mat applyMedianFilter5x5(const cv::Mat& image);

// Gaussian Filter
cv::Mat applyGaussianFilter3x3(const cv::Mat& image);
cv::Mat applyGaussianFilter5x5(const cv::Mat& image);

// Salt & Pepper Noise
cv::Mat applySaltPepperNoise02(const cv::Mat& image);
cv::Mat applySaltPepperNoise1(const cv::Mat& image);

// Rotation
cv::Mat applyRotationAttack15(const cv::Mat& image);
cv::Mat applyRotationAttack30(const cv::Mat& image);

// Scaling
cv::Mat applyScalingAttack05(const cv::Mat& image);
cv::Mat applyScalingAttack20(const cv::Mat& image);

// Translation
cv::Mat applyTranslationAttack10_10(const cv::Mat& image);
cv::Mat applyTranslationAttack20_40(const cv::Mat& image);

#endif // ATTACKS_H