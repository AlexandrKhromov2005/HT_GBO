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
std::vector<size_t> calcCoords(const std::vector<cv::Mat>& image_blocks);

//Function for embedding a bit
cv::Mat embedBit(cv::Mat block, double t, unsigned char w, unsigned char bit_wm);

//Function to extract a bit
unsigned char extractBit(cv::Mat& block, double t, unsigned char w);

