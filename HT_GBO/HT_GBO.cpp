#include <opencv2/opencv.hpp>
#include "src/affine.h"
#include "src/POB.h"
#include "src/image_processing.h"
#include "src/processWM.h"
#include <filesystem>
#include "src/POB.h"
#include "src/Hadamard.h"

cv::Mat generateRandomMatrix() {
    cv::Mat mat(4, 4, CV_8UC1);
    cv::randu(mat, cv::Scalar(0), cv::Scalar(255)); // Заполнение случайными значениями от 0 до 255
    return mat;
}


int main() {
    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;


    std::vector<int> keys = { 2, 1, -1, 3, 5, 4 };
    writeVectorToFile(keys);
    
    cv::Mat wm = importImage("images/pinguin.png");

    cv::Mat img = importImage("images/lenna_rgb.png");
   
    std::vector<unsigned char> check_vector;
    cv::Mat new_img = embedWatermark(img, wm, 52.9, check_vector);
    exportImage(new_img, "images/new_lenna.png");

    cv::Mat new_imp_img = importImage("images/new_lenna.png");
    cv::Mat extractedWatermark = extractWatermark(new_imp_img, 52.9, check_vector);
    exportImage(extractedWatermark, "images/new_wm.png");


    std::cout << "Done" << std::endl;
    std::cin.get();
    return 0;
}
