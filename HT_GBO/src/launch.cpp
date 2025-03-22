#include "launch.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

void embend_wm(const std::string& image, const std::string& new_image, const std::string& wm) {
    cv::Mat hostImage = importImage(image);
    cv::Mat watermark = importImage(wm);

    // Initialize GBO optimizer
    GBO gbo(hostImage, watermark);
    gbo.optimize();

    // Embed watermark with optimized threshold
    cv::Mat watermarkedImage = embedWatermark(hostImage, watermark, gbo.optimal_t);

    // Export the watermarked image
    exportImage(watermarkedImage, new_image);
}

void get_wm(const std::string& image, const std::string& new_wm) {
    cv::Mat watermarkedImage = importImage(image);
    cv::Mat extractedWM = extractWatermark(watermarkedImage, 30.0); // Default threshold

    // Export the extracted watermark
    exportImage(extractedWM, new_wm);
}

cv::Mat get_wm(const cv::Mat& cv_image) {
    return extractWatermark(cv_image, 30.0); // Default threshold
}

std::string getFileNameWithoutExtension(const std::string& path) {
    size_t lastSlashPos = path.find_last_of('/');
    size_t dotPos = path.find_last_of('.');

    if (lastSlashPos != std::string::npos && dotPos != std::string::npos) {
        return path.substr(lastSlashPos + 1, dotPos - lastSlashPos - 1);
    }

    return "";
}

void processAttack(
    const std::vector<cv::Mat>& embeddedImages,
    const cv::Mat& originalImage,
    const cv::Mat& originalWM,
    const AttackConfig& config,
    MetricCalculator metric,
    int iterations = 10,
    const std::string& output_file = ""
) {
    double mse_total = 0, psnr_total = 0, ncc_total = 0, ber_total = 0, ssim_total = 0;
    double max_mse = 0, max_psnr = 0, max_ncc = 0, max_ber = 0, max_ssim = 0;
    double min_mse = DBL_MAX, min_psnr = DBL_MAX, min_ncc = DBL_MAX, min_ber = DBL_MAX, min_ssim = DBL_MAX;

    std::ofstream output(output_file, std::ios::app);
    if (!output.is_open()) {
        std::cerr << "Error opening file: " << output_file << std::endl;
        return;
    }

    output << "Attack: " << config.name << std::endl;

    for (size_t i = 0; i < iterations; ++i) {
        cv::Mat attackedImg = config.attack(embeddedImages[i].clone());
        cv::Mat comparisonImg = config.use_cropped_comparison ? config.attack(originalImage.clone()) : originalImage.clone();
        cv::Mat extractedWM = extractWatermark(attackedImg, 30.0);

        double mse = metric(comparisonImg, attackedImg);
        double psnr = computeImagePSNR(comparisonImg, attackedImg);
        double ncc = computeImageNCC(comparisonImg, attackedImg);
        double ber = computeImageBER(originalWM, extractedWM);
        double ssim = computeImageSSIM(comparisonImg, attackedImg);

        // Update max and min values
        max_mse = std::max(max_mse, mse);
        min_mse = std::min(min_mse, mse);
        max_psnr = std::max(max_psnr, psnr);
        min_psnr = std::min(min_psnr, psnr);
        max_ncc = std::max(max_ncc, ncc);
        min_ncc = std::min(min_ncc, ncc);
        max_ber = std::max(max_ber, ber);
        min_ber = std::min(min_ber, ber);
        max_ssim = std::max(max_ssim, ssim);
        min_ssim = std::min(min_ssim, ssim);

        // Accumulate totals
        mse_total += mse;
        psnr_total += psnr;
        ncc_total += ncc;
        ber_total += ber;
        ssim_total += ssim;
    }

    // Write results to file
    output << "Average MSE: " << min_mse << " " << mse_total / iterations << " " << max_mse << std::endl
        << "Average PSNR: " << min_psnr << " " << psnr_total / iterations << " " << max_psnr << std::endl
        << "Average NCC: " << min_ncc << " " << ncc_total / iterations << " " << max_ncc << std::endl
        << "Average BER: " << min_ber << " " << ber_total / iterations << " " << max_ber << std::endl
        << "Average SSIM: " << min_ssim << " " << ssim_total / iterations << " " << max_ssim << std::endl
        << std::endl;

    output.close();
}

void launch(const std::string& image, const std::string& new_image, const std::string& wm, const std::string& new_wm) {
    std::vector<cv::Mat> embeddedImages;
    cv::Mat originalImage = importImage(image);
    cv::Mat originalWM = importImage(wm);

    // Embed watermark multiple times to get multiple embedded images
    for (size_t i = 0; i < 10; ++i) {
        embend_wm(image, new_image, wm);
        get_wm(new_image, new_wm);
        embeddedImages.push_back(importImage(new_image));
        std::cout << "\r" << i << "/10" << std::flush;
    }
    std::cout << "\r" << std::flush;

    // Define the attacks to test from test_attacks
    std::vector<AttackConfig> attacks = {
        {"No attack", [](const cv::Mat& img) { return img; }},
        {"Brightness Increase", [](const cv::Mat& img) { return brightnessIncrease(img, 10); }},
        {"Brightness Decrease", [](const cv::Mat& img) { return brightnessDecrease(img, 10); }},
        {"Contrast Increase", [](const cv::Mat& img) { return contrastIncrease(img, 1.5); }},
        {"Contrast Decrease", [](const cv::Mat& img) { return contrastDecrease(img, 0.5); }},
        {"Salt Pepper Noise", [](const cv::Mat& img) { return saltPepperNoise(img, 0.05); }},
        {"Speckle Noise", [](const cv::Mat& img) { return speckleNoise(img, 0.05); }},
        {"Histogram Equalization", [](const cv::Mat& img) { return histogramEqualization(img); }},
        {"Sharpening", [](const cv::Mat& img) { return sharpening(img); }},
        {"JPEG Compression (QF=90)", [](const cv::Mat& img) { return jpegCompression(img, 90); }},
        {"JPEG Compression (QF=80)", [](const cv::Mat& img) { return jpegCompression(img, 80); }},
        {"JPEG Compression (QF=70)", [](const cv::Mat& img) { return jpegCompression(img, 70); }},
        {"Gaussian Filtering", [](const cv::Mat& img) { return gaussianFiltering(img, 5); }},
        {"Median Filtering", [](const cv::Mat& img) { return medianFiltering(img, 5); }},
        {"Average Filtering", [](const cv::Mat& img) { return averageFiltering(img, 5); }},
        {"Cropping from Corner", [](const cv::Mat& img) { return cropFromCorner(img, 100); }, true},
        {"Cropping from Center", [](const cv::Mat& img) { return cropFromCenter(img, 100); }, true},
        {"Cropping from Edge", [](const cv::Mat& img) { return cropFromEdge(img, 100); }, true}
    };

    // Create results file name
    std::string resultFilename = "results_" + getFileNameWithoutExtension(image) + ".txt";
    std::ofstream resultsFile(resultFilename);
    resultsFile.close();

    // Process each attack
    for (const auto& attack : attacks) {
        MetricCalculator metric = computeImageMSE;
        processAttack(embeddedImages, originalImage, originalWM, attack, metric, 10, resultFilename);
    }
}