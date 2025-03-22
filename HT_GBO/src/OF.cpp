#include "OF.h"

inline double calc_omega(cv::Mat originalImage, cv::Mat watermarkedImage, cv::Mat originalWM, cv::Mat extractedWM) {
    double p = computePSNR(originalImage, watermarkedImage);
    double s = computeSSIM(originalImage, watermarkedImage);
    double nc = computeNC(originalWM, extractedWM);
    double ber = computeBER(originalWM, extractedWM);

    return p * s * nc * (1 - ber);
}

std::vector<WeightAttackPair> createAttacks(const cv::Mat& targetImage) {
    return {
        // JPEG Compression
        {0.7, [&]() { return applyJPEGCompression40(targetImage); }},
        {0.5, [&]() { return applyJPEGCompression70(targetImage); }},

        // JPEG2000 Compression
        {0.6, [&]() { return applyJPEG2000Compression41(targetImage); }},
        {0.4, [&]() { return applyJPEG2000Compression71(targetImage); }},

        // Median Filter
        {0.3, [&]() { return applyMedianFilter3x3(targetImage); }},
        {0.5, [&]() { return applyMedianFilter5x5(targetImage); }},

        // Gaussian Filter
        {0.4, [&]() { return applyGaussianFilter3x3(targetImage); }},
        {0.6, [&]() { return applyGaussianFilter5x5(targetImage); }},

        // Salt & Pepper Noise
        {0.3, [&]() { return applySaltPepperNoise02(targetImage); }},
        {0.5, [&]() { return applySaltPepperNoise1(targetImage); }},

        // Rotation
        {0.4, [&]() { return applyRotationAttack15(targetImage); }},
        {0.7, [&]() { return applyRotationAttack30(targetImage); }},

        // Scaling
        {0.6, [&]() { return applyScalingAttack05(targetImage); }},
        {0.5, [&]() { return applyScalingAttack20(targetImage); }},

        // Translation
        {0.3, [&]() { return applyTranslationAttack10_10(targetImage); }},
        {0.5, [&]() { return applyTranslationAttack20_40(targetImage); }}
    };
}

double objectiveFunction(std::vector<cv::Mat> originalImages, std::vector<cv::Mat> watermarkedImages,
    std::vector<cv::Mat> originalWMs, std::vector<cv::Mat> extractedWMs, double t) {
    double result = 0.0;
    int n = originalImages.size();
    int m = 16; // Number of attacks
    for (int i = 0; i < n; ++i) {
        double omega = calc_omega(originalImages[i], watermarkedImages[i], originalWMs[i], extractedWMs[i]);
        std::vector<WeightAttackPair> attackedImages = createAttacks(watermarkedImages[i]);
        for (int j = 0; j < m; ++j) {
            cv::Mat attackedImage = attackedImages[j].second();
            cv::Mat extractedWM = extractWatermark(attackedImage, t);
            double nc = computeNC(originalWMs[i], extractedWM);
            double ber = computeBER(originalWMs[i], extractedWM);
            result += (omega * attackedImages[j].first * nc * (1.0 - ber));
        }
    }
    return result / (n * m);
}