#include "Hadamard.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void Hadamard::applyHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F); // Конвертация в double

    // Матрица Адамара 4x4 (H4) из 1 и -1 (без нормировки)
    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1);

    // Одностороннее умножение: H4 * block (согласно уравнению 2)
    cv::Mat transformed = H4 * block_d;

    block = transformed.clone();
}

void Hadamard::applyInverseHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F);

    // Матрица H4 та же (H4 = H4^T для симметричной матрицы)
    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1);

    // Обратное преобразование: (H4 * transformed) / 4 (согласно уравнению 4)
    cv::Mat transformed = (H4 * block_d) / 4.0;

    // Корректное приведение к uchar с округлением
    transformed.convertTo(block, CV_8UC1);
}