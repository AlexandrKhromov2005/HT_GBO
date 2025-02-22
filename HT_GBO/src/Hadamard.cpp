#include "Hadamard.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Преобразование Адамара для матрицы 4x4
void Hadamard::applyHadamard(cv::Mat& block) {
    // Матрица Адамара размерности 4x4
    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1) / 2.0;

    // Умножаем блок на матрицу Адамара
    block = H4 * block;
}

// Умножение матрицы 4x4 на матрицу Адамара
cv::Mat Hadamard::multiplyWithHadamard(cv::Mat& input) {
    cv::Mat result = input.clone();
    applyHadamard(result);
    return result;
}

// Обратное преобразование Адамара для матрицы 4x4
void Hadamard::applyInverseHadamard(cv::Mat& block) {
    // Транспонированная матрица Адамара для 4x4
    cv::Mat H4_T = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1) / 2.0;

    // Умножаем на транспонированную матрицу Адамара (обратное преобразование)
    block = H4_T * block;
}

