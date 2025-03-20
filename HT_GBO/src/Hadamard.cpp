#include "Hadamard.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void Hadamard::applyHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F); // ����������� � double

    // ������� ������� 4x4 (H4) �� 1 � -1 (��� ����������)
    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1);

    // ������������� ���������: H4 * block (�������� ��������� 2)
    cv::Mat transformed = H4 * block_d;

    block = transformed.clone();
}

void Hadamard::applyInverseHadamard(cv::Mat& block) {
    cv::Mat block_d;
    block.convertTo(block_d, CV_64F);

    // ������� H4 �� �� (H4 = H4^T ��� ������������ �������)
    cv::Mat H4 = (cv::Mat_<double>(4, 4) <<
        1, 1, 1, 1,
        1, -1, 1, -1,
        1, 1, -1, -1,
        1, -1, -1, 1);

    // �������� ��������������: (H4 * transformed) / 4 (�������� ��������� 4)
    cv::Mat transformed = (H4 * block_d) / 4.0;

    // ���������� ���������� � uchar � �����������
    transformed.convertTo(block, CV_8UC1);
}