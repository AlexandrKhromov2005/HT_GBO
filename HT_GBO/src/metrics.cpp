#include "metrics.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cmath>

double computePSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);       
    s1.convertTo(s1, CV_64F);     
    s1 = s1.mul(s1);              

    cv::Scalar s = cv::sum(s1);    

    double sse = s.val[0] + s.val[1] + s.val[2]; 
    if (sse <= 1e-10) 
        return 100.0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

double computeSSIM(const cv::Mat& I1, const cv::Mat& I2)
{
    const double C1 = 6.5025, C2 = 58.5225;

    int d = CV_32F;

    cv::Mat I1_, I2_;
    I1.convertTo(I1_, d);
    I2.convertTo(I2_, d);

    cv::Mat I1_2 = I1_.mul(I1_);
    cv::Mat I2_2 = I2_.mul(I2_);
    cv::Mat I1_I2 = I1_.mul(I2_);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1_, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2_, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);  

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);   

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);
    double ssimValue = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
    return ssimValue;
}

double computeNC(const cv::Mat& watermarkOriginal, const cv::Mat& watermarkExtracted)
{
    cv::Mat origFloat, extFloat;
    watermarkOriginal.convertTo(origFloat, CV_32F);
    watermarkExtracted.convertTo(extFloat, CV_32F);

    double dotProduct = origFloat.dot(extFloat);
    double normOrig = cv::norm(origFloat);
    double normExt = cv::norm(extFloat);
    if (normOrig == 0 || normExt == 0)
        return 0;
    double nc = dotProduct / (normOrig * normExt);
    return nc;
}

double computeBER(const cv::Mat& binaryImage1, const cv::Mat& binaryImage2)
{
    CV_Assert(binaryImage1.size() == binaryImage2.size());
    int errorCount = 0;
    int totalPixels = binaryImage1.rows * binaryImage1.cols;

    for (int i = 0; i < binaryImage1.rows; i++)
    {
        for (int j = 0; j < binaryImage1.cols; j++)
        {
            uchar a = binaryImage1.at<uchar>(i, j) > 128 ? 1 : 0;
            uchar b = binaryImage2.at<uchar>(i, j) > 128 ? 1 : 0;
            if (a != b)
                errorCount++;
        }
    }
    return static_cast<double>(errorCount) / totalPixels;
}
