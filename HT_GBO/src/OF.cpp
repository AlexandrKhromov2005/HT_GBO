#include "OF.h"


//inline double calc_omega(cv::Mat& originalImage, cv::Mat& watermarkedImage, cv::Mat& originalWM, cv::Mat& extractedWM) { //here with no attack due to article
//	double p = computePSNR(originalImage, watermarkedImage);
//	double s = computeSSIM(originalImage, watermarkedImage);
//	double nc = computeNC(originalWM, extractedWM); 
//	double ber = computeBER(originalWM, extractedWM); 
//
//	return p * s * nc * (1 - ber);
//}
//
//std::vector<WeightAttackPair> createAttacks(const cv::Mat& targetImage) {
//    return {
//        // JPEG Compression
//        {0.8, [&]() { return applyJPEGCompression70(targetImage); }},
//        {0.6, [&]() { return applyJPEGCompression80(targetImage); }},
//        {0.4, [&]() { return applyJPEGCompression90(targetImage); }},
//
//        // Median Filter
//        {0.3, [&]() { return applyMedianFilter3(targetImage); }},
//        {0.5, [&]() { return applyMedianFilter5(targetImage); }},
//        {0.7, [&]() { return applyMedianFilter7(targetImage); }},
//
//        // Gaussian Filter
//        {0.4, [&]() { return applyGaussianFilter3(targetImage); }},
//        {0.6, [&]() { return applyGaussianFilter5(targetImage); }},
//        {0.8, [&]() { return applyGaussianFilter7(targetImage); }},
//
//        // Salt & Pepper Noise
//        {0.3, [&]() { return applySaltPepperNoise1(targetImage); }},
//        {0.6, [&]() { return applySaltPepperNoise5(targetImage); }},
//        {0.9, [&]() { return applySaltPepperNoise10(targetImage); }},
//
//        // Rotation
//        {0.4, [&]() { return applyRotationAttack10(targetImage); }},
//        {0.7, [&]() { return applyRotationAttack45(targetImage); }},
//        {0.9, [&]() { return applyRotationAttack90(targetImage); }},
//
//        // Scaling
//        {0.6, [&]() { return applyScalingAttack05(targetImage); }},
//        {0.5, [&]() { return applyScalingAttack15(targetImage); }},
//        {0.7, [&]() { return applyScalingAttack20(targetImage); }},
//
//        // Translation
//        {0.3, [&]() { return applyTranslationAttack10(targetImage); }},
//        {0.5, [&]() { return applyTranslationAttack20(targetImage); }},
//        {0.7, [&]() { return applyTranslationAttack30(targetImage); }}
//    };
//}
//
//
//inline double objectiveFunction(const std::vector<cv::Mat&> originalImages, std::vector<cv::Mat&> watermarkedImages, const std::vector<cv::Mat&> originalWMs, const std::vector<cv::Mat&> extractedWMs) {
//    double result = 0.0;
//    int n = originalImages.size();
//    int m = 7 * 3;
//    for (int i = 0; i < n; ++i) {
//        double omega = calc_omega(originalImages[i], watermarkedImages[i], originalWMs[i], extractedWMs[i]);
//        std::vector<WeightAttackPair> attackedImages = createAttacks(originalImages[i]);
//        for (int j = 0; j < m; ++j) {
//            cv::Mat extractedWM; //Tagir, u need to add here method for extracting WM
//            double nc = computeNC(originalWMs[i], extractedWM);
//            double ber = computeBER(originalWMs[i], extractedWM);
//            result += (omega * attackedImages[i].first * nc * ber);
//        }
//    }
//    return result;
//}
