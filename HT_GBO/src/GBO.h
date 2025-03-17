#ifndef GBO_H
#define GBO_H

#include <opencv2/opencv.hpp>
#include "population.h"

class GBO {
public:
    const cv::Mat& hostImage;
    const cv::Mat& wm;
    double optimal_t;

    GBO(const cv::Mat& host, const cv::Mat& watermark)
        : hostImage(host), wm(watermark), optimal_t(30.0) {
    }

    void optimize();
};

#endif // GBO_H