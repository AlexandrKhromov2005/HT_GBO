#include "image_processing.h"
#include "Hadamard.h"

// Function to export image (safe version without compression parameter)
bool exportImage(const cv::Mat& image, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot export empty image");
    }

    cv::Mat outputImage;
    try {
        // Конвертируем цветовое пространство из RGB в BGR
        cv::cvtColor(image, outputImage, cv::COLOR_RGB2BGR);
    }
    catch (const cv::Exception& e) {
        throw std::runtime_error("Color conversion failed: " + std::string(e.what()));
    }

    // Параметры сохранения PNG без сжатия (0 – отсутствие сжатия)
    std::vector<int> compression_params = { cv::IMWRITE_PNG_COMPRESSION, 0 };

    try {
        if (!cv::imwrite(outputPath, outputImage, compression_params)) {
            throw std::runtime_error("Failed to write image data to filesystem");
        }
    }
    catch (const cv::Exception& e) {
        throw std::runtime_error("Image write operation failed: " + std::string(e.what()));
    }

    return true;
}

// Function to import image (safe version)
cv::Mat importImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath +
            " (check file existence and permissions)");
    }

    try {
        cv::Mat convertedImage;
        cv::cvtColor(image, convertedImage, cv::COLOR_BGR2RGB);
        return convertedImage;
    }
    catch (const cv::Exception& e) {
        throw std::runtime_error("Color space conversion failed: " + std::string(e.what()));
    }
}



// Function to split image into blockSize x blockSize blocks
std::vector<cv::Mat> splitIntoBlocks(const cv::Mat& image) {
    int blockSize = 4;

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
cv::Mat assembleImage(const std::vector<cv::Mat>& blocks, int originalRows, int originalCols) {
    int blockSize = 4;

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
std::vector<size_t> calcCoords(const cv::Mat& hostImage, KEY_B& key_b) {
    int blockSize = 4;
    std::vector<cv::Mat> image_blocks = splitIntoBlocks(hostImage);
    std::vector<size_t> coords;
    for (size_t i = 0; i < image_blocks.size(); i++) {
        std::pair<int, int> block_data = {i, key_b[i / 16]};
        std::string hash = computeMD5(block_data);
        
        if (hash[0] != 'a'){
            coords.push_back(i);
        }
    }
    return coords;
}

void print_block(cv::Mat block) {
    std::cout << "\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << block.at<double>(i, j) << " ";
        }
        std::cout << "\n";
    }
}

// Function to embed one bit into block using frequency domain quantization
cv::Mat embedBit(cv::Mat block, double t, unsigned char mode, unsigned char bit_wm) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F);
    Hadamard hadamardUtil;
    hadamardUtil.applyHadamard(block_d);

    if (mode == 1) {
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
    else { 
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
    hadamardUtil.applyHadamard(block_d);
    unsigned int sum = 0;
    if (mode == 1) {
        for (int i = 0; i < 3; i++) {
            double value = block_d.at<double>(0, i);
            int quant = static_cast<int>(std::trunc(value / t));
            unsigned char bit = quant % 2;
            sum += bit;
        }
        return (sum >= 2) ? 1 : 0;
    }
    else { 
        double value = block_d.at<double>(0, 0);
        int quant = static_cast<int>(std::trunc(value / t));
        return quant % 2;
    }
}

// Function to embed watermark into host image
cv::Mat embedWatermarkLayer(const cv::Mat& hostImage, const cv::Mat& wm, double t, unsigned char layer) {
    std::vector<std::pair<int, int>> encrypted_wm = process(wm);
    KEY_B key_b = get();

    int blockSize = 4;
    size_t requiredBlocks = wm.rows * wm.cols * 15;
    std::vector<cv::Mat> blocks = splitIntoBlocks(hostImage);

    std::vector<size_t> coords = calcCoords(hostImage, key_b);
    if (coords.size() < requiredBlocks) {
        std::cerr << "Not enough blocks available for embedding watermark!" << std::endl;
        return cv::Mat();
    }
    for (int i = 0; i < WM_SIZE * WM_SIZE; i++) {
        unsigned char wm_pixel = wm.at<uchar>(i);
        unsigned char upper_bits = encrypted_wm[i].first; 
        unsigned char lower_bits = encrypted_wm[i].second;        

        for (int j = 0; j < 4; j++) {
            unsigned char bit = (upper_bits >> j + 4) & 1;
                size_t block_index = coords[i * 7 + j];
                blocks[block_index] = embedBit(blocks[block_index], t, 1, bit);
        }

        for (int j = 0; j < 3; j++) {
            size_t block_index = coords[i * 7 + 4 + j];
            unsigned char bit = (lower_bits >> j) & 1;
            blocks[block_index] = embedBit(blocks[block_index], t, 2, bit);
        }
    }
    return assembleImage(blocks, hostImage.rows, hostImage.cols);
}

// Function to extract watermark from watermarked image
cv::Mat extractWatermarkLayer(const cv::Mat& watermarkedImage, double t, unsigned char layer) {
    KEY_B key_b = get();

    int blockSize = 4;
    size_t requiredBlocks = WM_SIZE * WM_SIZE * 15;
    std::vector<cv::Mat> blocks = splitIntoBlocks(watermarkedImage);

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
        for (int j = 0; j < 4; j++) {
            unsigned int sum = 0;
            size_t block_index = coords[i * 7 + j];
            unsigned char bit = extractBit(blocks[block_index], t, 1);
            upper_bits |= (bit << j + 4);
        }
        for (int j = 0; j < 3; j++) {
            size_t block_index = coords[i * 7 + 4 + j];
            unsigned char bit = extractBit(blocks[block_index], t, 2);
            lower_compressed |= (bit << j);
        }

        std::pair<int, int> pixel = { upper_bits, lower_compressed };
        pixels.push_back(pixel);
    }
    extractedWM = restore(pixels, layer);

    return extractedWM;
}


// Function to embed watermark into host image
cv::Mat embedWatermark(const cv::Mat& hostImage, const cv::Mat& wm, double t) {
    std::string file_name_key_b = "keys/KEY_B.txt";

    std::ofstream outFile_key_b(file_name_key_b);
    outFile_key_b.close();

    std::vector<cv::Mat> img_channels;
    cv::split(hostImage, img_channels);

    std::vector<cv::Mat> wm_channels;
    cv::split(wm, wm_channels);

    img_channels[0] = embedWatermarkLayer(img_channels[0], wm_channels[0], t, 0);
    img_channels[1] = embedWatermarkLayer(img_channels[1], wm_channels[1], t, 1);
    img_channels[2] = embedWatermarkLayer(img_channels[2], wm_channels[2], t, 2);

    cv::Mat newImage = cv::Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC3);
    cv::merge(img_channels, newImage);

    return newImage;
}

// Function to extract watermark from watermarked image
cv::Mat extractWatermark(const cv::Mat& watermarkedImage, double t) {
    std::vector<cv::Mat> img_channels;
    cv::split(watermarkedImage, img_channels);

    cv::Mat wm = cv::Mat::zeros(WM_SIZE, WM_SIZE, CV_8UC3);
    std::vector<cv::Mat> wm_channels;
    cv::split(wm, wm_channels);

    wm_channels[0] = extractWatermarkLayer(img_channels[0], t, 0);
    wm_channels[1] = extractWatermarkLayer(img_channels[1], t, 1);
    wm_channels[2] = extractWatermarkLayer(img_channels[2], t, 2);

    cv::merge(wm_channels, wm);

    return wm;
}

