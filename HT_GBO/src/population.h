#ifndef POPULATION_H
#define POPULATION_H

#include <array>
#include <utility>
#include "config.h"
#include "random_utils.h"
#include <opencv2/opencv.hpp>
#include "OF.h"

using VecOf = std::pair<double, double>; // (t, OF_value)

class Population {
public:
    std::array<VecOf, POP_SIZE> vecs;
    VecOf worst_vec;
    size_t best_ind;

    Population();
    void initOf(const cv::Mat& hostImage, const cv::Mat& wm);
    double calculateOf(const cv::Mat& hostImage, const cv::Mat& wm, double t);
    void update(VecOf trial, size_t vec_ind);
};

#endif // POPULATION_H