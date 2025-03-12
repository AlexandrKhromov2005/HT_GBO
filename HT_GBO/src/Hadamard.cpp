#include "Hadamard.h"
#include <opencv2/opencv.hpp>
#include <iostream>


void Hadamard::applyHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F); 

    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1) / 2.0;

    cv::Mat transformed = H4 * block_d * H4.t();

    block = transformed.clone(); 
}

void Hadamard::applyInverseHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F); 

    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1) / 2.0;

    cv::Mat transformed = H4 * block_d * H4.t();

    
    block = transformed.clone();
    block.forEach<double>([](double& pixel, const int* position) {
        pixel = cv::saturate_cast<uchar>(std::round(pixel));
        });
}


