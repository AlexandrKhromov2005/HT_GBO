#ifndef AFFINE_H
#define AFFINE_H

#include <opencv2/opencv.hpp>
#include <utility>
#include "config.h"

cv::Mat affineTransform(cv::Mat wm, int a, int b, int c, int d, int tx, int ty);
cv::Mat affineTransformInv(cv::Mat wm, int a, int b, int c, int d, int tx, int ty);

#endif //AFFINE_H