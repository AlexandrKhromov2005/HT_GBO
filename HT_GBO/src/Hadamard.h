#ifndef HADAMARD_H
#define HADAMARD_H

#include <opencv2/opencv.hpp>

class Hadamard {
public:
    // Преобразование Адамара для матрицы размерности 4x4
    static void applyHadamard(cv::Mat& block);

    // Умножение матрицы на матрицу Адамара для размера 4x4
    static cv::Mat multiplyWithHadamard(cv::Mat& input);

    // Обратное преобразование Адамара
    static void applyInverseHadamard(cv::Mat& block);
};

#endif // HADAMARD_H
