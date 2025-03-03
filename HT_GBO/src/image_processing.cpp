#include "image_processing.h"
#include "Hadamard.h"
#include "POB.h"

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
        std::cerr << "Error creating context EVP_MD_CTX" << std::endl;
        return "";
    }

    if (1 != EVP_DigestInit_ex(ctx, EVP_md5(), nullptr)) {
        std::cerr << "Initialization error EVP для MD5" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    if (1 != EVP_DigestUpdate(ctx, block.data, block.total())) {
        std::cerr << "Error in EVP_DigestUpdate" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    if (1 != EVP_DigestFinal_ex(ctx, result, &resultLen)) {
        std::cerr << "Error in EVP_DigestFinal_ex" << std::endl;
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

// Embed a single bit into a block using quantization
cv::Mat embedBit(cv::Mat block, double t, unsigned char mode, unsigned char bit_wm) {
    if (mode == 1) {
        for (int i = 0; i < 3; i++) {
            double value = static_cast<double>(block.at<uchar>(i));
            int quant = static_cast<int>(std::round(value / t));
            unsigned char e_cond = (quant % 2) ^ bit_wm;
            if (e_cond == 0) {
                block.at<uchar>(i) = static_cast<uchar>(quant * t + t / 3);
            }
            else {
                block.at<uchar>(i) = static_cast<uchar>(quant * t - t / 3);
            }
        }
    }
    else { // mode == 2
        double value = static_cast<double>(block.at<uchar>(0));
        int quant = static_cast<int>(std::round(value / t));
        unsigned char e_cond = (quant % 2) ^ bit_wm;
        if (e_cond == 0) {
            block.at<uchar>(0) = static_cast<uchar>(quant * t + t / 3);
        }
        else {
            block.at<uchar>(0) = static_cast<uchar>(quant * t - t / 3);
        }
    }
    return block;
}

// Extract a single bit from a block
unsigned char extractBit(const cv::Mat& block, double t, unsigned char mode) {
    unsigned int sum = 0;
    if (mode == 1) {
        for (int i = 0; i < 3; i++) {
            double value = static_cast<double>(block.at<uchar>(i));
            int quant = static_cast<int>(std::round(value / t));
            unsigned char bit = quant % 2;
            sum += bit;
        }
        return (sum >= 2) ? 1 : 0;
    }
    else { // mode == 2
        double value = static_cast<double>(block.at<uchar>(0));
        int quant = static_cast<int>(std::round(value / t));
        return quant % 2;
    }
}

// Embed a watermark into the image
cv::Mat embedWatermark(const cv::Mat& hostImage, const cv::Mat& wm, double t, std::vector<size_t> coords) {
    int blockSize = 4;
    std::vector<cv::Mat> blocks = splitIntoBlocks(hostImage, blockSize);
    size_t requiredBlocks = wm.rows * wm.cols * 7;

    if (coords.size() < requiredBlocks) {
        std::cerr << "Not enough blocks available for embedding watermark!" << std::endl;
        return cv::Mat();
    }

    Hadamard hadamardUtil;

    for (int i = 0; i < wm.rows * wm.cols; i++) {
        unsigned char wm_pixel = wm.at<uchar>(i);
        unsigned char upper_bits = (wm_pixel >> 4) & 0x0F;
        unsigned char lower_bits = wm_pixel & 0x0F;

        // Embed upper 4 bits
        for (int j = 0; j < 4; j++) {
            size_t block_index = coords[i * 7 + j];
            hadamardUtil.applyHadamard(blocks[block_index]);
            unsigned char bit = (upper_bits >> j) & 1;
            blocks[block_index] = embedBit(blocks[block_index], t, 1, bit);
            hadamardUtil.applyInverseHadamard(blocks[block_index]);
        }

        // Compress lower 4 bits using POB
        std::pair<int, int> pob_lower = pob(lower_bits);

        // Embed 3 bits of the compressed value
        for (int j = 0; j < 3; j++) {
            size_t block_index = coords[i * 7 + j + 4];
            hadamardUtil.applyHadamard(blocks[block_index]);
            unsigned char bit = (pob_lower.first >> j) & 1;
            blocks[block_index] = embedBit(blocks[block_index], t, 2, bit);
            hadamardUtil.applyInverseHadamard(blocks[block_index]);
        }
    }

    return assembleImage(blocks, hostImage.rows, hostImage.cols, blockSize);
}

// Extract a watermark from the watermarked image
cv::Mat extractWatermark(const cv::Mat& watermarkedImage, int wm_rows, int wm_cols, double t, std::vector<size_t> coords) {
    int blockSize = 4;
    std::vector<cv::Mat> blocks = splitIntoBlocks(watermarkedImage, blockSize);
    size_t requiredBlocks = wm_rows * wm_cols * 7;

    if (coords.size() < requiredBlocks) {
        std::cerr << "Not enough blocks available for extracting watermark!" << std::endl;
        return cv::Mat();
    }

    Hadamard hadamardUtil;
    cv::Mat extractedWM(wm_rows, wm_cols, CV_8UC1);

    for (int i = 0; i < wm_rows * wm_cols; i++) {
        unsigned char upper_bits = 0;
        unsigned char lower_compressed = 0;

        // Extract upper 4 bits
        for (int j = 0; j < 4; j++) {
            size_t block_index = coords[i * 7 + j];
            cv::Mat block = blocks[block_index].clone();
            hadamardUtil.applyHadamard(block);
            unsigned char bit = extractBit(block, t, 1);
            hadamardUtil.applyInverseHadamard(block);
            upper_bits |= (bit << j);
        }

        // Extract compressed 3 bits
        for (int j = 0; j < 3; j++) {
            size_t block_index = coords[i * 7 + j + 4];
            cv::Mat block = blocks[block_index].clone();
            hadamardUtil.applyHadamard(block);
            unsigned char bit = extractBit(block, t, 2);
            hadamardUtil.applyInverseHadamard(block);
            lower_compressed |= (bit << j);
        }

        // Recover lower 4 bits
        int r_fixed = 2;
        unsigned char lower_bits = static_cast<unsigned char>(inverse_pob(lower_compressed, r_fixed) & 0x0F);
        extractedWM.at<uchar>(i) = (upper_bits << 4) | lower_bits;
    }

    return extractedWM;
}