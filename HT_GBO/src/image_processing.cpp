#include "image_processing.h"
#include "Hadamard.h"
#include "POB.h"

// Function to import image (grayscale)
cv::Mat importImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image: " << imagePath << std::endl;
    }
    return image;
}

// Function to split image into blockSize x blockSize blocks
std::vector<cv::Mat> splitIntoBlocks(const cv::Mat& image, int blockSize) {
    std::vector<cv::Mat> blocks;
    if (image.rows < blockSize || image.cols < blockSize) {
        std::cerr << "Image is too small for " << blockSize << "x" << blockSize << " blocks" << std::endl;
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

// Function to assemble image from blocks
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

// Function to export image
bool exportImage(const cv::Mat& image, const std::string& outputPath) {
    if (image.empty()) {
        std::cerr << "Empty image for export" << std::endl;
        return false;
    }
    std::vector<int> compression_params = { cv::IMWRITE_PNG_COMPRESSION, 9 };
    bool success = cv::imwrite(outputPath, image, compression_params);
    if (!success) {
        std::cerr << "Failed to save image to: " << outputPath << std::endl;
    }
    return success;
}

std::string computeMD5(const std::pair<int, int>& pair) {
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int resultLen = 0;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        std::cerr << "Error creating EVP_MD_CTX" << std::endl;
        return "";
    }

    if (1 != EVP_DigestInit_ex(ctx, EVP_md5(), nullptr)) {
        std::cerr << "EVP_DigestInit_ex error" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    if (1 != EVP_DigestUpdate(ctx, &pair, sizeof(pair))) {
        std::cerr << "EVP_DigestUpdate error" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    if (1 != EVP_DigestFinal_ex(ctx, result, &resultLen)) {
        std::cerr << "EVP_DigestFinal_ex error" << std::endl;
        EVP_MD_CTX_free(ctx);
        return "";
    }

    EVP_MD_CTX_free(ctx);

    std::ostringstream hashString;
    for (unsigned int i = 0; i < resultLen; ++i) {
        hashString << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(result[i]);
    }

    return hashString.str();
}

// Function to compute embedding coordinates (simplified: taking all block indices here)
std::vector<size_t> calcCoords(const cv::Mat& hostImage, KEY_B key_b) {
    int blockSize = 4;
    std::vector<cv::Mat> image_blocks = splitIntoBlocks(hostImage, blockSize);
    std::vector<size_t> coords;
    std::cout << "blocks num = " << image_blocks.size() << "\n";
    std::cout << "r num = " << key_b.size() << "\n";
    for (size_t i = 0; i < image_blocks.size(); i++) {
        std::pair<int, int> block_data = {i, key_b[i / 16]};
        std::string hash = computeMD5(block_data);
        
        if (hash[0] != 'a'){
            coords.push_back(i);
        }
    }
    return coords;
}

// Function to embed one bit into block using frequency domain quantization
cv::Mat embedBit(cv::Mat block, double t, unsigned char mode, unsigned char bit_wm) {
    // Convert block to double format for precise calculations
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F);
    Hadamard hadamardUtil;
    // Forward Hadamard transform (convert to frequency domain)
    hadamardUtil.applyHadamard(block_d);

    if (mode == 1) {
        // Mode 1: embed bit into first 3 coefficients of the first row
        for (int i = 0; i < 3; i++) {
            double value = block_d.at<double>(0, i);
            int quant = static_cast<int>(std::round(value / t));
            unsigned char e_cond = (quant % 2) ^ bit_wm;
            if (e_cond == 0) {
                block_d.at<double>(0, i) = quant * t + t / 3.0;
            }
            else {
                block_d.at<double>(0, i) = quant * t - t / 3.0;
            }
        }
    }
    else { // mode == 2: embedding for lower bits
        double value = block_d.at<double>(0, 0);
        int quant = static_cast<int>(std::round(value / t));
        unsigned char e_cond = (quant % 2) ^ bit_wm;
        if (e_cond == 0) {
            block_d.at<double>(0, 0) = quant * t + t / 3.0;
        }
        else {
            block_d.at<double>(0, 0) = quant * t - t / 3.0;
        }
    }
    // Inverse Hadamard transform (convert back to spatial domain)
    hadamardUtil.applyInverseHadamard(block_d);
    cv::Mat block_out;
    block_d.convertTo(block_out, CV_8UC1);
    return block_out;
}

// Function to extract one bit from block
unsigned char extractBit(const cv::Mat& block, double t, unsigned char mode) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F);
    Hadamard hadamardUtil;
    // Forward Hadamard transform to convert to frequency domain
    hadamardUtil.applyHadamard(block_d);
    unsigned int sum = 0;
    if (mode == 1) {
        // Extraction from first 3 coefficients with voting
        for (int i = 0; i < 3; i++) {
            double value = block_d.at<double>(0, i);
            int quant = static_cast<int>(std::trunc(value / t));
            unsigned char bit = quant % 2;
            sum += bit;
        }
        return (sum >= 2) ? 1 : 0;
    }
    else { // mode == 2
        double value = block_d.at<double>(0, 0);
        int quant = static_cast<int>(std::trunc(value / t));
        return quant % 2;
    }
}

// Function to embed watermark into host image
cv::Mat embedWatermark(const cv::Mat& hostImage, const cv::Mat& wm, double t) {
    std::vector<std::pair<int, int>> encrypted_wm = process(wm);
    KEY_B key_b = get();

    int blockSize = 4;
    // Required number of blocks = watermark pixels * 15 (12 for upper bits + 3 for lower bits after POB)
    size_t requiredBlocks = wm.rows * wm.cols * 15;
    std::vector<cv::Mat> blocks = splitIntoBlocks(hostImage, blockSize);

    std::vector<size_t> coords = calcCoords(hostImage, key_b);
    if (coords.size() < requiredBlocks) {
        std::cerr << "Not enough blocks available for embedding watermark!" << std::endl;
        return cv::Mat();
    }
    for (int i = 0; i < wm.rows * wm.cols; i++) {
        unsigned char wm_pixel = wm.at<uchar>(i);
        unsigned char upper_bits = encrypted_wm[i].first; // Upper 4 bits
        unsigned char lower_bits = encrypted_wm[i].second;        // Lower 4 bits

        // Embed upper 4 bits: each bit embedded into 3 blocks (for robustness - voting)
        for (int j = 0; j < 4; j++) {
            unsigned char bit = (upper_bits >> j) & 1;
            for (int k = 0; k < 3; k++) {
                size_t block_index = coords[i * 15 + j * 3 + k];
                blocks[block_index] = embedBit(blocks[block_index], t, 1, bit);
            }
        }

        // Compress lower 4 bits using POB (pob() function implemented in POB.h)
        std::pair<int, int> pob_lower = pob(lower_bits);
        // Embed compressed lower bits (assumed 3-bit representation) into 3 blocks
        for (int j = 0; j < 3; j++) {
            size_t block_index = coords[i * 15 + 12 + j];
            unsigned char bit = (pob_lower.first >> j) & 1;
            blocks[block_index] = embedBit(blocks[block_index], t, 2, bit);
        }
    }
    return assembleImage(blocks, hostImage.rows, hostImage.cols, blockSize);
}

// Function to extract watermark from watermarked image
cv::Mat extractWatermark(const cv::Mat& watermarkedImage, double t) {
    KEY_B key_b = get();

    int blockSize = 4;
    size_t requiredBlocks = WM_SIZE * WM_SIZE * 15;
    std::vector<cv::Mat> blocks = splitIntoBlocks(watermarkedImage, blockSize);

    std::vector<size_t> coords = calcCoords(watermarkedImage, key_b);
    if (coords.size() < requiredBlocks) {
        std::cerr << "Not enough blocks available for extracting watermark!" << std::endl;
        return cv::Mat();
    }
    cv::Mat extractedWM(WM_SIZE, WM_SIZE, CV_8UC1);

    std::vector<std::pair<int, int>> pixels;
    for (int i = 0; i < WM_SIZE * WM_SIZE; i++) {
        unsigned char upper_bits = 0;
        unsigned char lower_compressed = 0;
        // Extract upper 4 bits from 12 blocks (3 blocks per bit) with voting
        for (int j = 0; j < 4; j++) {
            unsigned int sum = 0;
            for (int k = 0; k < 3; k++) {
                size_t block_index = coords[i * 15 + j * 3 + k];
                unsigned char bit = extractBit(blocks[block_index], t, 1);
                sum += bit;
            }
            unsigned char bit_extracted = (sum >= 2) ? 1 : 0;
            upper_bits |= (bit_extracted << j);
        }
        // Extract compressed lower bits from 3 blocks
        for (int j = 0; j < 3; j++) {
            size_t block_index = coords[i * 15 + 12 + j];
            unsigned char bit = extractBit(blocks[block_index], t, 2);
            lower_compressed |= (bit << j);
        }

        std::pair<int, int> pixel;
        pixel.first = upper_bits;
        pixel.second = lower_compressed;

        pixels.push_back(pixel);
    }
    extractedWM = restore(pixels);

    return extractedWM;
}
