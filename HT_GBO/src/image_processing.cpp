#include "image_processing.h"

// Image import function
cv::Mat importImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
    }
    return image;
}

// Function to split image into 4x4 blocks
std::vector<cv::Mat> splitIntoBlocks(const cv::Mat& image, int blockSize) {
    std::vector<cv::Mat> blocks;

    if (image.rows < blockSize || image.cols < blockSize) {
        std::cerr << "Image is too small for 4x4 blocks" << std::endl;
        return blocks;
    }

    for (int y = 0; y <= image.rows - blockSize; y += blockSize) {
        for (int x = 0; x <= image.cols - blockSize; x += blockSize) {
            cv::Rect roi(x, y, blockSize, blockSize);
            blocks.push_back(image(roi).clone());
        }
    }

    return blocks;
}

// Function for assembling an image from blocks
cv::Mat assembleImage(const std::vector<cv::Mat>& blocks, int originalRows, int originalCols, int blockSize) {
    cv::Mat result(originalRows, originalCols, CV_8UC1);

    int blockIndex = 0;
    for (int y = 0; y <= originalRows - blockSize; y += blockSize) {
        for (int x = 0; x <= originalCols - blockSize; x += blockSize) {
            if (blockIndex < blocks.size()) {
                cv::Rect roi(x, y, blockSize, blockSize);
                blocks[blockIndex++].copyTo(result(roi));
            }
        }
    }

    return result;
}

// Image export function
bool exportImage(const cv::Mat& image, const std::string& outputPath) {
    if (image.empty()) {
        std::cerr << "Empty image for export" << std::endl;
        return false;
    }

    std::vector<int> compression_params = { cv::IMWRITE_JPEG_QUALITY, 95 };
    bool success = cv::imwrite(outputPath, image, compression_params);

    if (!success) {
        std::cerr << "Failed to save image to: " << outputPath << std::endl;
    }
    return success;
}
// Function to calculate MD5 from 4x4 block via EVP
std::string computeMD5(const cv::Mat& block) {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int resultLen = 0;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        std::cerr << "Ошибка создания контекста EVP_MD_CTX" << std::endl;
        return "";
    }

    if (1 != EVP_DigestInit_ex(ctx, EVP_md5(), nullptr)) {
        std::cerr << "Ошибка инициализации EVP для MD5" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    if (1 != EVP_DigestUpdate(ctx, block.data, block.total())) {
        std::cerr << "Ошибка в EVP_DigestUpdate" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    if (1 != EVP_DigestFinal_ex(ctx, result, &resultLen)) {
        std::cerr << "Ошибка в EVP_DigestFinal_ex" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    EVP_MD_CTX_free(ctx);

    std::ostringstream hashString;
    for (unsigned int i = 0; i < resultLen; ++i) {
        hashString << std::hex << std::setw(2) << std::setfill('0') << (int)result[i];
    }

    return hashString.str();
}

//Function for calculating embedding coordinates
std::vector<size_t> calcCoords(const std::vector<cv::Mat>& image_blocks) {
    std::string hash;
    std::vector<size_t> coords;

    for (size_t i = 0; i < image_blocks.size(); i++) {
        hash = computeMD5(image_blocks[i]);
        if (hash[0] == 'a' || hash[0] == 'b' || hash[0] == 'c' || hash[0] == 'd' || hash[0] == 'e' || hash[0] == 'f') {
            coords.push_back(i);
        }
    }

    return coords;
}

//Function for embedding a bit
cv::Mat embendBit(cv::Mat block, double t, unsigned char w, unsigned char bit_wm) {
    if (w == 1) {
        for (int i = 0; i < 3; i++) {
            unsigned char e_cond = (static_cast<int>(round(block.at<uchar>(i) / t)) % 2) ^ bit_wm;
            if (e_cond == 0) {
                block.at<uchar>(i) = static_cast<unsigned char>(static_cast<int>(round(block.at<uchar>(i) / t)) * t + t / 3);
            }
            else {
                block.at<uchar>(i) = static_cast<unsigned char>(static_cast<int>(round(block.at<uchar>(i) / t)) * t - t / 3);
            }
        }
    }
    else {
        unsigned char e_cond = (static_cast<int>(round(block.at<uchar>(0) / t)) % 2) ^ bit_wm;
        if (e_cond == 0) {
            block.at<uchar>(0) = static_cast<unsigned char>(static_cast<int>(round(block.at<uchar>(0) / t)) * t + t / 3);
        }
        else {
            block.at<uchar>(0) = static_cast<unsigned char>(static_cast<int>(round(block.at<uchar>(0) / t)) * t - t / 3);
        }
    }
    return block;
}

//Function to extract a bit
unsigned char extract_bit(cv::Mat& block, double t, unsigned char w, unsigned char bit_wm) {
    int temp_bit = 0;
    if (w == 1) {
        for (int i = 0; i < 3; i++) {
            temp_bit = static_cast<int>(trunc(block.at<double>(i) / t)) % 2;
        }
        if (temp_bit >= 2) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else {
        temp_bit = static_cast<int>(trunc(block.at<double>(0) / t)) % 2;
        return temp_bit;
    }
}

