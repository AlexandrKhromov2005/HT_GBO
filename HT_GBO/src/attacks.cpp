#include "attacks.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

// Helper functions with parameters (internal only)
namespace {
    cv::Mat applyJPEGCompression(const cv::Mat& image, int quality) {
        std::vector<uchar> buf;
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };
        cv::imencode(".jpg", image, buf, params);
        return cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    }

        cv::Mat applyMedianFilter(const cv::Mat& image, int kernelSize) {
        cv::Mat result;
        cv::medianBlur(image, result, kernelSize);
        return result;
    }

    cv::Mat applyGaussianFilter(const cv::Mat& image, int kernelSize, double sigma) {
        cv::Mat result;
        cv::GaussianBlur(image, result, cv::Size(kernelSize, kernelSize), sigma);
        return result;
    }

    cv::Mat applySaltPepperNoise(const cv::Mat& image, double noiseRatio) {
        cv::Mat noisy = image.clone();
        int nNoisePixels = static_cast<int>(noiseRatio * image.rows * image.cols);
        std::default_random_engine generator;
        std::uniform_int_distribution<int> rowDist(0, image.rows - 1);
        std::uniform_int_distribution<int> colDist(0, image.cols - 1);
        for (int i = 0; i < nNoisePixels; ++i) {
            int r = rowDist(generator);
            int c = colDist(generator);
            if (image.channels() == 1) {
                noisy.at<uchar>(r, c) = (rand() % 2) ? 255 : 0;
            }
            else {
                for (int ch = 0; ch < image.channels(); ch++) {
                    noisy.at<cv::Vec3b>(r, c)[ch] = (rand() % 2) ? 255 : 0;
                }
            }
        }
        return noisy;
    }

    cv::Mat applyRotationAttack(const cv::Mat& image, double angle) {
        cv::Point2f center(image.cols / 2.0F, image.rows / 2.0F);
        cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat rotated;
        cv::warpAffine(image, rotated, rotMat, image.size());
        return rotated;
    }

    cv::Mat applyScalingAttack(const cv::Mat& image, double scaleFactor) {
        cv::Mat scaled;
        cv::resize(image, scaled, cv::Size(), scaleFactor, scaleFactor);
        cv::Mat restored;
        cv::resize(scaled, restored, image.size());
        return restored;
    }

    cv::Mat applyTranslationAttack(const cv::Mat& image, int offsetX, int offsetY) {
        cv::Mat transMat = (cv::Mat_<double>(2, 3) << 1, 0, offsetX, 0, 1, offsetY);
        cv::Mat translated;
        cv::warpAffine(image, translated, transMat, image.size());
        return translated;
    }

}

// JPEG Compression implementations
cv::Mat applyJPEGCompression70(const cv::Mat& image) { return applyJPEGCompression(image, 70); }
cv::Mat applyJPEGCompression80(const cv::Mat& image) { return applyJPEGCompression(image, 80); }
cv::Mat applyJPEGCompression90(const cv::Mat& image) { return applyJPEGCompression(image, 90); }

// Median Filter implementations
cv::Mat applyMedianFilter3(const cv::Mat& image) { return applyMedianFilter(image, 3); }
cv::Mat applyMedianFilter5(const cv::Mat& image) { return applyMedianFilter(image, 5); }
cv::Mat applyMedianFilter7(const cv::Mat& image) { return applyMedianFilter(image, 7); }

// Gaussian Filter implementations
cv::Mat applyGaussianFilter3(const cv::Mat& image) { return applyGaussianFilter(image, 3, 0.5); }
cv::Mat applyGaussianFilter5(const cv::Mat& image) { return applyGaussianFilter(image, 5, 1.0); }
cv::Mat applyGaussianFilter7(const cv::Mat& image) { return applyGaussianFilter(image, 7, 1.5); }

// Salt & Pepper implementations
cv::Mat applySaltPepperNoise1(const cv::Mat& image) { return applySaltPepperNoise(image, 0.01); }
cv::Mat applySaltPepperNoise5(const cv::Mat& image) { return applySaltPepperNoise(image, 0.05); }
cv::Mat applySaltPepperNoise10(const cv::Mat& image) { return applySaltPepperNoise(image, 0.1); }

// Rotation implementations
cv::Mat applyRotationAttack10(const cv::Mat& image) { return applyRotationAttack(image, 10.0); }
cv::Mat applyRotationAttack45(const cv::Mat& image) { return applyRotationAttack(image, 45.0); }
cv::Mat applyRotationAttack90(const cv::Mat& image) { return applyRotationAttack(image, 90.0); }

// Scaling implementations
cv::Mat applyScalingAttack05(const cv::Mat& image) { return applyScalingAttack(image, 0.5); }
cv::Mat applyScalingAttack15(const cv::Mat& image) { return applyScalingAttack(image, 1.5); }
cv::Mat applyScalingAttack20(const cv::Mat& image) { return applyScalingAttack(image, 2.0); }

// Translation implementations
cv::Mat applyTranslationAttack10(const cv::Mat& image) { return applyTranslationAttack(image, 10, 10); }
cv::Mat applyTranslationAttack20(const cv::Mat& image) { return applyTranslationAttack(image, 20, 20); }
cv::Mat applyTranslationAttack30(const cv::Mat& image) { return applyTranslationAttack(image, 30, 30); }