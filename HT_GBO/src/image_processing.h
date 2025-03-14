#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <openssl/evp.h>

#include <opencv2/opencv.hpp>
#include <vector>

// Image import function
cv::Mat importImage(const std::string& imagePath);

// Function to split image into 4x4 blocks
std::vector<cv::Mat> splitIntoBlocks(const cv::Mat& image, int blockSize = 4);

// Function for assembling an image from blocks
cv::Mat assembleImage(const std::vector<cv::Mat>& blocks, int originalRows, int originalCols, int blockSize = 4);

// Image export function
bool exportImage(const cv::Mat& image, const std::string& outputPath);

// Function to calculate MD5 from 4x4 block via EVP
std::string computeMD5(const cv::Mat& block);

//Function for calculating embedding coordinates
std::vector<size_t> calcCoords(const cv::Mat& hostImage);

//Function for embedding a bit
cv::Mat embedBit(cv::Mat block, double t, unsigned char mode, unsigned char bit_wm);

// Extract a single bit from a block
unsigned char extractBit(const cv::Mat& block, double t, unsigned char mode);

// Embed a watermark into the image
cv::Mat embedWatermark(const cv::Mat& hostImage, const cv::Mat& wm, double t, std::vector<size_t> coords, std::vector<int>& r_vec);

// Extract a watermark from the watermarked image
cv::Mat extractWatermark(const cv::Mat& watermarkedImage, int wm_rows, int wm_cols, double t, std::vector<size_t> coords, std::vector<int> r_vec);
