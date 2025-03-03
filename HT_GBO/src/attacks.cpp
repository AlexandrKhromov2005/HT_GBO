#include "attacks.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

cv::Mat applyJPEGCompression(const cv::Mat& image, int quality)
{
    std::vector<uchar> buf;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, quality };
    cv::imencode(".jpg", image, buf, params);
    cv::Mat compressed = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    return compressed;
}

cv::Mat applyMedianFilter(const cv::Mat& image, int kernelSize)
{
    cv::Mat result;
    cv::medianBlur(image, result, kernelSize);
    return result;
}

cv::Mat applyGaussianFilter(const cv::Mat& image, int kernelSize, double sigma)
{
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(kernelSize, kernelSize), sigma);
    return result;
}

cv::Mat applySaltPepperNoise(const cv::Mat& image, double noiseRatio)
{
    cv::Mat noisy = image.clone();
    int nNoisePixels = static_cast<int>(noiseRatio * image.rows * image.cols);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> rowDist(0, image.rows - 1);
    std::uniform_int_distribution<int> colDist(0, image.cols - 1);
    for (int i = 0; i < nNoisePixels; ++i)
    {
        int r = rowDist(generator);
        int c = colDist(generator);
        if (image.channels() == 1)
        {
            noisy.at<uchar>(r, c) = (rand() % 2) ? 255 : 0;
        }
        else 
        {
            for (int ch = 0; ch < image.channels(); ch++)
            {
                noisy.at<cv::Vec3b>(r, c)[ch] = (rand() % 2) ? 255 : 0;
            }
        }
    }
    return noisy;
}

cv::Mat applyRotationAttack(const cv::Mat& image, double angle)
{
    cv::Point2f center(image.cols / 2.0F, image.rows / 2.0F);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, rotMat, image.size());
    return rotated;
}

cv::Mat applyScalingAttack(const cv::Mat& image, double scaleFactor)
{
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scaleFactor, scaleFactor);
    cv::Mat restored;
    cv::resize(scaled, restored, image.size());
    return restored;
}

cv::Mat applyTranslationAttack(const cv::Mat& image, int offsetX, int offsetY)
{
    cv::Mat transMat = (cv::Mat_<double>(2, 3) << 1, 0, offsetX, 0, 1, offsetY);
    cv::Mat translated;
    cv::warpAffine(image, translated, transMat, image.size());
    return translated;
}
