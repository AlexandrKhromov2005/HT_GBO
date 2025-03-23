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
        1, -1, -1, 1);

    cv::Mat transformed = H4 * block_d;

    block = transformed.clone();
}

void Hadamard::applyInverseHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F);

    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1);

    cv::Mat transformed = (H4 * block_d) / 4.0;

    transformed.convertTo(block, CV_8UC1);
}

