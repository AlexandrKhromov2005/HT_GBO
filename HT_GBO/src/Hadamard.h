#ifndef HADAMARD_H
#define HADAMARD_H

#include <opencv2/opencv.hpp>

class Hadamard {
public:
    // �������������� ������� ��� ������� ����������� 4x4
    static void applyHadamard(cv::Mat& block);

    // ��������� ������� �� ������� ������� ��� ������� 4x4
    static cv::Mat multiplyWithHadamard(cv::Mat& input);

    // �������� �������������� �������
    static void applyInverseHadamard(cv::Mat& block);
};

#endif // HADAMARD_H
