#include "processWM.h"

KEY_A read() {
    std::ifstream file("keys/KEY_A");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    KEY_A result;
    for (int i = 0; i < 6; ++i) {
        if (!(file >> result[i])) {
            throw std::runtime_error("File contains invalid data or not enough numbers");
        }
    }

    return result;
}

void update(int n) {
    std::ofstream file("keys/KEY_B", std::ios::app);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing");
    }

    file << n << '\n';

    if (file.fail()) {
        throw std::runtime_error("Failed to write number to file");
    }
}

KEY_B get() {
    std::ifstream file("keys/KEY_B");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    KEY_B result;

    for (int i = 0; i < WM_SIZE * WM_SIZE; ++i) {
        if (!(file >> result[i])) {
            throw std::runtime_error("File contains invalid data or not enough numbers");
        }
    }
}

cv::Mat process(cv::Mat wm) {
    KEY_A key_a = read();
    cv::Mat affined_wm = affineTransform(wm, key_a[0], key_a[1], key_a[2], key_a[3], key_a[4], key_a[5]);

    //POB 
    for (int x = 0; x < affined_wm.cols; ++x) {
        for (int y = 0; y < affined_wm.rows; ++y) {
            int temp_pixel = affined_wm.at<uchar>(x, y);
            int high_bits = temp_pixel & 0b11110000;
            int low_bits = temp_pixel & 0b00001111;

            std::pair<int, int> p = pob(low_bits);
            low_bits = p.first;
            update(p.second);

            temp_pixel = high_bits | low_bits;

            affined_wm.at<uchar>(x, y) = temp_pixel;
        }
    }

    return affined_wm;
}

cv::Mat restore(cv::Mat wm) {
    KEY_B key_b = get();
    KEY_A key_a = read();
    int cnt = 0;
    cv::Mat restored_wm = cv::Mat::zeros(wm.rows, wm.cols,  CV_8UC1);

    for (int x = 0; x < wm.cols; ++x) {
        for (int y = 0; y < wm.rows; ++y) {
            int temp_pixel = wm.at<uchar>(x, y);
            int high_bits = temp_pixel & 0b11110000;
            int low_bits = temp_pixel & 0b00001111;

            low_bits = inverse_pob(low_bits, key_b[cnt]);
            ++cnt;

            temp_pixel = high_bits | low_bits;
            restored_wm.at<uchar>(x, y) = temp_pixel;
        }
    }

    return restored_wm;
}