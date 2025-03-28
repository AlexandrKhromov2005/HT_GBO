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

    cv::Mat applyJPEG2000Compression(const cv::Mat& image, double compressionRatio) {
        std::vector<uchar> buf;
        std::vector<int> params = { cv::IMWRITE_JPEG2000_COMPRESSION_X1000, static_cast<int>(compressionRatio * 1000) };
        cv::imencode(".jp2", image, buf, params);
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
                cv::Vec3b& pixel = noisy.at<cv::Vec3b>(r, c);
                bool white = (rand() % 2);
                pixel[0] = white ? 255 : 0; // Blue
                pixel[1] = white ? 255 : 0; // Green
                pixel[2] = white ? 255 : 0; // Red
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
cv::Mat applyJPEGCompression40(const cv::Mat& image) { return applyJPEGCompression(image, 40); }
cv::Mat applyJPEGCompression70(const cv::Mat& image) { return applyJPEGCompression(image, 70); }

// JPEG2000 Compression implementations
cv::Mat applyJPEG2000Compression41(const cv::Mat& image) { return applyJPEG2000Compression(image, 4.1); }
cv::Mat applyJPEG2000Compression71(const cv::Mat& image) { return applyJPEG2000Compression(image, 7.1); }

// Median Filter implementations
cv::Mat applyMedianFilter3x3(const cv::Mat& image) { return applyMedianFilter(image, 3); }
cv::Mat applyMedianFilter5x5(const cv::Mat& image) { return applyMedianFilter(image, 5); }

// Gaussian Filter implementations
cv::Mat applyGaussianFilter3x3(const cv::Mat& image) { return applyGaussianFilter(image, 3, 0.5); }
cv::Mat applyGaussianFilter5x5(const cv::Mat& image) { return applyGaussianFilter(image, 5, 1.0); }

// Salt & Pepper Noise implementations
cv::Mat applySaltPepperNoise02(const cv::Mat& image) { return applySaltPepperNoise(image, 0.02); }
cv::Mat applySaltPepperNoise1(const cv::Mat& image) { return applySaltPepperNoise(image, 0.01); }

// Rotation implementations
cv::Mat applyRotationAttack15(const cv::Mat& image) { return applyRotationAttack(image, 15.0); }
cv::Mat applyRotationAttack30(const cv::Mat& image) { return applyRotationAttack(image, 30.0); }

// Scaling implementations
cv::Mat applyScalingAttack05(const cv::Mat& image) { return applyScalingAttack(image, 0.5); }
cv::Mat applyScalingAttack20(const cv::Mat& image) { return applyScalingAttack(image, 2.0); }

// Translation implementations
cv::Mat applyTranslationAttack10_10(const cv::Mat& image) { return applyTranslationAttack(image, 10, 10); }
cv::Mat applyTranslationAttack20_40(const cv::Mat& image) { return applyTranslationAttack(image, 20, 40); }