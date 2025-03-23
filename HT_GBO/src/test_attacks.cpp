#include "test_attacks.h"

// A1: ������ JPEG � ��������� 40
cv::Mat jpegCompression40(const cv::Mat& img) {
    std::vector<uchar> buffer;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 40 };
    cv::imencode(".jpg", img, buffer, params);
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

// A2: ������ JPEG � ��������� 70
cv::Mat jpegCompression70(const cv::Mat& img) {
    std::vector<uchar> buffer;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 70 };
    cv::imencode(".jpg", img, buffer, params);
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

// A5: ��������� ������ 3x3
cv::Mat medianFilter3x3(const cv::Mat& img) {
    cv::Mat result;
    cv::medianBlur(img, result, 3);
    return result;
}

// A6: ��������� ������ 5x5
cv::Mat medianFilter5x5(const cv::Mat& img) {
    cv::Mat result;
    cv::medianBlur(img, result, 5);
    return result;
}

// A7: ����������� ������ 3x3
cv::Mat gaussianFilter3x3(const cv::Mat& img) {
    cv::Mat result;
    cv::GaussianBlur(img, result, cv::Size(3, 3), 0);
    return result;
}

// A8: ����������� ������ 5x5
cv::Mat gaussianFilter5x5(const cv::Mat& img) {
    cv::Mat result;
    cv::GaussianBlur(img, result, cv::Size(5, 5), 0);
    return result;
}

// A9: ��� 0.2%
cv::Mat saltPepperNoise02(const cv::Mat& img) {
    cv::Mat result = img.clone();
    int n = 0.002 * img.total();

    for (int i = 0; i < n; i++) {
        int row = rand() % img.rows;
        int col = rand() % img.cols;
        result.at<cv::Vec3b>(row, col) = (rand() % 2) ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
    }
    return result;
}

// A10: ��� 1%
cv::Mat saltPepperNoise1(const cv::Mat& img) {
    cv::Mat result = img.clone();
    int n = 0.01 * img.total();

    for (int i = 0; i < n; i++) {
        int row = rand() % img.rows;
        int col = rand() % img.cols;
        result.at<cv::Vec3b>(row, col) = (rand() % 2) ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
    }
    return result;
}

// AA: ������� �� 15� � �������
cv::Mat rotate15(const cv::Mat& img) {
    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 15, 1.0);
    cv::Mat rotated;
    cv::warpAffine(img, rotated, rot, img.size());
    cv::warpAffine(rotated, rotated, cv::getRotationMatrix2D(center, -15, 1.0), img.size());
    return rotated;
}

// AB: ������� �� 30� � �������
cv::Mat rotate30(const cv::Mat& img) {
    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, 30, 1.0);
    cv::Mat rotated;
    cv::warpAffine(img, rotated, rot, img.size());
    cv::warpAffine(rotated, rotated, cv::getRotationMatrix2D(center, -30, 1.0), img.size());
    return rotated;
}

// AC: ���������� 0.5x � �������
cv::Mat scale05(const cv::Mat& img) {
    cv::Mat scaled;
    cv::resize(img, scaled, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::resize(scaled, scaled, img.size(), 0, 0, cv::INTER_LINEAR);
    return scaled;
}

// AD: ���������� 4x � �������
cv::Mat scale4(const cv::Mat& img) {
    cv::Mat scaled;
    cv::resize(img, scaled, cv::Size(), 4.0, 4.0, cv::INTER_LINEAR);
    cv::resize(scaled, scaled, img.size(), 0, 0, cv::INTER_LINEAR);
    return scaled;
}

// AE: ����� (10,10) � �������
cv::Mat translate10(const cv::Mat& img) {
    cv::Mat trans = (cv::Mat_<double>(2, 3) << 1, 0, 10, 0, 1, 10);
    cv::Mat translated;
    cv::warpAffine(img, translated, trans, img.size());
    trans.at<double>(0, 2) = -10;
    trans.at<double>(1, 2) = -10;
    cv::warpAffine(translated, translated, trans, img.size());
    return translated;
}

// AF: ����� (20,40) � �������
cv::Mat translate20_40(const cv::Mat& img) {
    cv::Mat trans = (cv::Mat_<double>(2, 3) << 1, 0, 20, 0, 1, 40);
    cv::Mat translated;
    cv::warpAffine(img, translated, trans, img.size());
    trans.at<double>(0, 2) = -20;
    trans.at<double>(1, 2) = -40;
    cv::warpAffine(translated, translated, trans, img.size());
    return translated;
}


