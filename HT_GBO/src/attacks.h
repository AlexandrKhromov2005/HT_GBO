#ifndef ATTACKS_H
#define ATTACKS_H

#include <opencv2/opencv.hpp>

// JPEG Compression
cv::Mat applyJPEGCompression70(const cv::Mat& image);
cv::Mat applyJPEGCompression80(const cv::Mat& image);
cv::Mat applyJPEGCompression90(const cv::Mat& image);

// Median Filter
cv::Mat applyMedianFilter3(const cv::Mat& image);
cv::Mat applyMedianFilter5(const cv::Mat& image);
cv::Mat applyMedianFilter7(const cv::Mat& image);

// Gaussian Filter
cv::Mat applyGaussianFilter3(const cv::Mat& image);
cv::Mat applyGaussianFilter5(const cv::Mat& image);
cv::Mat applyGaussianFilter7(const cv::Mat& image);

// Salt & Pepper Noise
cv::Mat applySaltPepperNoise1(const cv::Mat& image);
cv::Mat applySaltPepperNoise5(const cv::Mat& image);
cv::Mat applySaltPepperNoise10(const cv::Mat& image);

// Rotation
cv::Mat applyRotationAttack10(const cv::Mat& image);
cv::Mat applyRotationAttack45(const cv::Mat& image);
cv::Mat applyRotationAttack90(const cv::Mat& image);

// Scaling
cv::Mat applyScalingAttack05(const cv::Mat& image);
cv::Mat applyScalingAttack15(const cv::Mat& image);
cv::Mat applyScalingAttack20(const cv::Mat& image);

// Translation
cv::Mat applyTranslationAttack10(const cv::Mat& image);
cv::Mat applyTranslationAttack20(const cv::Mat& image);
cv::Mat applyTranslationAttack30(const cv::Mat& image);

#endif // ATTACKS_H