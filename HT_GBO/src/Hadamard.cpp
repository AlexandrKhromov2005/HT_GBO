#include "Hadamard.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// �������������� ������� ��� ������� 4x4
void Hadamard::applyHadamard(cv::Mat& block) {
    // ������� ������� ����������� 4x4
    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1) / 2.0;

    // �������� ���� �� ������� �������
    block = H4 * block;
}

// ��������� ������� 4x4 �� ������� �������
cv::Mat Hadamard::multiplyWithHadamard(cv::Mat& input) {
    cv::Mat result = input.clone();
    applyHadamard(result);
    return result;
}

// �������� �������������� ������� ��� ������� 4x4
void Hadamard::applyInverseHadamard(cv::Mat& block) {
    // ����������������� ������� ������� ��� 4x4
    cv::Mat H4_T = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1) / 2.0;

    // �������� �� ����������������� ������� ������� (�������� ��������������)
    block = H4_T * block;
}

