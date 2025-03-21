#include "population.h"
#include "image_processing.h"

Population::Population() {
    for (size_t i = 0; i < POP_SIZE; ++i) {
        vecs[i].first = 30.0 + 30.0 * rand_num(); // t ∈ [30, 60]
        vecs[i].second = DBL_MAX;
    }
    best_ind = 0;
    worst_vec.first = vecs[0].first;
    worst_vec.second = -DBL_MAX;
}

double Population::calculateOf(const cv::Mat& hostImage, const cv::Mat& wm, double t) {
    cv::Mat watermarked = embedWatermark(hostImage, wm, t);
    cv::Mat extractedWM = extractWatermark(watermarked, t);

    std::vector<cv::Mat> origImgs = { hostImage };
    std::vector<cv::Mat> wmImgs = { watermarked };
    std::vector<cv::Mat> origWMs = { wm };
    std::vector<cv::Mat> exWMs = { extractedWM };

    return objectiveFunction(origImgs, wmImgs, origWMs, exWMs, t);
}

void Population::initOf(const cv::Mat& hostImage, const cv::Mat& wm) {
    double ofbest = -DBL_MAX;
    double ofworst = DBL_MAX;

    for (size_t i = 0; i < POP_SIZE; ++i) {
        vecs[i].second = calculateOf(hostImage, wm, vecs[i].first);
        if (vecs[i].second > ofbest) {
            ofbest = vecs[i].second;
            best_ind = i;
        }
        if (vecs[i].second < ofworst) {
            ofworst = vecs[i].second;
            worst_vec = vecs[i];
        }
    }
}

void Population::update(VecOf trial, size_t vec_ind) {
    if (trial.second > vecs[vec_ind].second) { // Чем больше OF, тем лучше
        vecs[vec_ind] = trial;
        if (vecs[vec_ind].second > vecs[best_ind].second) {
            best_ind = vec_ind;
        }
    }
    else if (vecs[vec_ind].second < worst_vec.second) {
        worst_vec = vecs[vec_ind];
    }
}