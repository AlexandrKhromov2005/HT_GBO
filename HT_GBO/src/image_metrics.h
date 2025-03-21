#ifndef IMAGE_METRICS_H
#define IMAGE_METRICS_H

#include <opencv2/opencv.hpp>

// ¬ычисление среднеквадратичной ошибки (MSE) между двум€ изображени€ми
double computeImageMSE(const cv::Mat& image1, const cv::Mat& image2);

// ¬ычисление отношени€ пикового сигнала к шуму (PSNR) между двум€ изображени€ми
double computeImagePSNR(const cv::Mat& image1, const cv::Mat& image2);

// ¬ычисление битовой ошибки (BER) дл€ бинарных изображений
double computeImageBER(const cv::Mat& image1, const cv::Mat& image2);

// ¬ычисление нормированной коррел€ции (NCC) между двум€ изображени€ми
double computeImageNCC(const cv::Mat& img1, const cv::Mat& img2);

double computeImageSSIM(const cv::Mat& img1, const cv::Mat& img2);


#endif // IMAGE_METRICS_H
