#include "image_metrics.h"
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

double computeImageMSE(const cv::Mat& image1, const cv::Mat& image2) {
    if (image1.size() != image2.size() || image1.type() != image2.type()) {
        std::cerr << "Error: Images must have the same size and type." << std::endl;
        return -1;
    }

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    cv::Scalar mse_scalar = cv::mean(diff);
    double mse = (mse_scalar[0] + mse_scalar[1] + mse_scalar[2]) / 3.0;

    return mse;
}

double computeImagePSNR(const cv::Mat& image1, const cv::Mat& image2) {
    double mse = computeImageMSE(image1, image2);
    if (mse <= 0) {
        return 0;
    }

    double max_pixel_value = 255.0;
    double psnr = 10.0 * log10((max_pixel_value * max_pixel_value) / mse);
    return psnr;
}

double computeImageBER(const cv::Mat& image1, const cv::Mat& image2) {
    if (image1.size() != image2.size() || image1.type() != CV_8UC3 || image2.type() != CV_8UC3) {
        std::cerr << "Error: Images must have the same size and be RGB (CV_8UC3)." << std::endl;
        return -1.0;
    }

    std::vector<cv::Mat> channels1, channels2;
    cv::split(image1, channels1);
    cv::split(image2, channels2);

    double total_errors = 0.0;
    for (int c = 0; c < 3; ++c) {
        cv::Mat bin1, bin2;
        cv::threshold(channels1[c], bin1, 127, 1, cv::THRESH_BINARY);
        cv::threshold(channels2[c], bin2, 127, 1, cv::THRESH_BINARY);

        cv::Mat diff;
        cv::absdiff(bin1, bin2, diff);
        total_errors += cv::countNonZero(diff);
    }

    int total_bits = 3 * image1.rows * image1.cols;
    return total_errors / total_bits;
}

double computeImageNCC(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != CV_8UC3 || img2.type() != CV_8UC3) {
        std::cerr << "Error: Images must have the same size and be RGB (CV_8UC3)." << std::endl;
        return -1.0;
    }

    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);

    cv::Scalar mean1, stddev1, mean2, stddev2;
    cv::meanStdDev(img1f, mean1, stddev1);
    cv::meanStdDev(img2f, mean2, stddev2);

    // ѕроверка нулевого стандартного отклонени€
    for (int c = 0; c < 3; ++c) {
        if (stddev1[c] == 0 || stddev2[c] == 0) {
            std::cerr << "Warning: Zero standard deviation in channel " << c << std::endl;
            return -1.0;
        }
    }

    cv::Mat norm1 = (img1f - mean1) / stddev1;
    cv::Mat norm2 = (img2f - mean2) / stddev2;

    cv::Mat product = norm1.mul(norm2);
    cv::Scalar ncc_per_channel = cv::mean(product);

    return (ncc_per_channel[0] + ncc_per_channel[1] + ncc_per_channel[2]) / 3.0;
}

double computeImageSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "Error: Images must have the same size and type." << std::endl;
        return -1;
    }

    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);
    double C1 = 6.5025, C2 = 58.5225;

    cv::Mat mu1, mu2, sigma1_2, sigma2_2, sigma12;
    GaussianBlur(img1f, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(img2f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    GaussianBlur(img1f.mul(img1f), sigma1_2, cv::Size(11, 11), 1.5);
    GaussianBlur(img2f.mul(img2f), sigma2_2, cv::Size(11, 11), 1.5);
    GaussianBlur(img1f.mul(img2f), sigma12, cv::Size(11, 11), 1.5);

    sigma1_2 -= mu1_2;
    sigma2_2 -= mu2_2;
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = mu1_2 + mu2_2 + C1;
    cv::Mat t4 = sigma1_2 + sigma2_2 + C2;

    cv::Mat ssim_map;
    divide(t1.mul(t2), t3.mul(t4), ssim_map);

    cv::Scalar ssim_scalar = mean(ssim_map);
    return (ssim_scalar[0] + ssim_scalar[1] + ssim_scalar[2]) / 3.0;  
}
