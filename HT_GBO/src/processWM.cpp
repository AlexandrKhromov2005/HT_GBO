#include "processWM.h"
#include <bitset>

std::filesystem::path get_exe_directory() {
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
}

KEY_A read() {
    auto exe_dir = get_exe_directory();
    std::filesystem::path file_path = exe_dir / "keys/KEY_A.txt";

    std::ifstream file(file_path);
    if (!file) {
        //std::cerr << "Failed to open file: " << file_path << std::endl;
    }
    //std::cout << "File opened successfully: " << file_path << std::endl;

    KEY_A result;
    for (int i = 0; i < 6; ++i) {
        if (!(file >> result[i])) {
            throw std::runtime_error("File contains invalid data or not enough numbers");
        }
    }

    return result;
}

void update(int n) {
    std::ofstream file("keys/KEY_B.txt", std::ios::app);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing");
    }

    file << n << '\n';

    if (file.fail()) {
        throw std::runtime_error("Failed to write number to file");
    }
}

KEY_B get() {
    auto exe_dir = get_exe_directory();
    std::filesystem::path file_path = exe_dir / "keys/KEY_B.txt";

    std::ifstream file(file_path);
    if (!file) {
        //std::cerr << "Failed to open file: " << file_path << std::endl;
    }
    //std::cout << "File opened successfully: " << file_path << std::endl;

    KEY_B result;
    int num;
    while (file >> num) {
        result.push_back(num);
    }

    if (file.bad()) {
        throw std::runtime_error("Error reading file");
    }   

    return result;
}

std::vector<std::pair<int, int>> process(cv::Mat wm) {
    KEY_A key_a = read();
    cv::Mat affined_wm = affineTransform(wm, key_a[0], key_a[1], key_a[2], key_a[3], key_a[4], key_a[5]);

    //POB 
    std::vector<std::pair<int, int>> result;

    for (int x = 0; x < affined_wm.cols; ++x) {
        for (int y = 0; y < affined_wm.rows; ++y) {
            int temp_pixel = affined_wm.at<uchar>(x, y);
            int high_bits = temp_pixel & 0b11110000;
            int low_bits = temp_pixel & 0b00001111;

            std::pair<int, int> p = pob(low_bits);
            low_bits = p.first;
            update(p.second);

            std::pair<int, int> bits_pair = { high_bits, low_bits };
            result.push_back(bits_pair);
        }
    }

    return result;
}

cv::Mat restore(std::vector<std::pair<int, int>> pixels, unsigned char layer) {
    KEY_B key_b = get();
    KEY_A key_a = read();
    int cnt = 0;
    cv::Mat restored_wm = cv::Mat::zeros(WM_SIZE, WM_SIZE, CV_8UC1);

    for (int x = 0; x < WM_SIZE; ++x) {
        for (int y = 0; y < WM_SIZE; ++y) {
            unsigned char high_bits = pixels[cnt].first;
            unsigned char low_bits_comp = pixels[cnt].second;

            unsigned char low_bits = inverse_pob(low_bits_comp, key_b[layer * 1024 + cnt]);
            ++cnt;

            unsigned char temp_pixel = high_bits | low_bits;

            restored_wm.at<uchar>(x, y) = temp_pixel;
        }
    }

    restored_wm = affineTransformInv(restored_wm, key_a[0], key_a[1], key_a[2], key_a[3], key_a[4], key_a[5]);

    return restored_wm;
}

bool writeVectorToFile(const std::vector<int>& keys) {
    std::string file_name_key_a = "keys/KEY_A.txt";
    std::string file_name_key_b = "keys/KEY_B.txt";

    std::ofstream outFile_key_a(file_name_key_a);
    std::ofstream outFile_key_b(file_name_key_b);
    outFile_key_b.close();

    if (!outFile_key_a.is_open()) {
        return false; // Ошибка открытия файла
    }

    // Записываем элементы через пробел
    if (!keys.empty()) {
        outFile_key_a << keys[0]; // Первый элемент без пробела
        for (size_t i = 1; i < keys.size(); ++i) {
            outFile_key_a << " " << keys[i]; // Остальные элементы с пробелом
        }
    }

    // Проверяем флаги ошибок
    const bool success = !outFile_key_a.fail();
    outFile_key_a.close();

    return success;
}