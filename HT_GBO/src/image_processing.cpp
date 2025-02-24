#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <openssl/evp.h>
#include <sstream>
#include <iomanip>
#include <cmath>

// Функция для обработки изображения и разбиения на блоки 4x4
std::vector<cv::Mat> processImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Ошибка загрузки изображения!" << std::endl;
        return {};
    }

    std::cout << "Размер изображения: " << image.rows << "x" << image.cols << std::endl;

    int blockSize = 4;
    int rows = image.rows;
    int cols = image.cols;

    if (rows < blockSize || cols < blockSize) {
        std::cerr << "Изображение слишком маленькое для блоков 4x4." << std::endl;
        return {};
    }

    std::vector<cv::Mat> blocks;

    for (int row = 0; row <= rows - blockSize; row += blockSize) {
        for (int col = 0; col <= cols - blockSize; col += blockSize) {
            cv::Mat block = image(cv::Rect(col, row, blockSize, blockSize)).clone();
            blocks.push_back(block);
        }
    }

    return blocks;
}

// Функция для восстановления изображения из блоков 4x4
cv::Mat reconstructImage(const std::vector<cv::Mat>& blocks, int rows, int cols, int blockSize = 4) {
    cv::Mat reconstructedImage(rows, cols, CV_8UC1);

    int blockIdx = 0;
    for (int row = 0; row <= rows - blockSize; row += blockSize) {
        for (int col = 0; col <= cols - blockSize; col += blockSize) {
            if (blockIdx < blocks.size()) {
                blocks[blockIdx++].copyTo(reconstructedImage(cv::Rect(col, row, blockSize, blockSize)));
            }
        }
    }

    return reconstructedImage;
}

// Функция для вычисления MD5 от блока 4x4 через EVP
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

//Функция для вычисления координат встраивания
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

//Функция для встраивания бита
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

//Функция для извлечения бита
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

