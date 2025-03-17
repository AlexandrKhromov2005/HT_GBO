#include <opencv2/opencv.hpp>
#include "src/affine.h"
#include "src/POB.h"
#include "src/image_processing.h"
#include "src/processWM.h"
#include <filesystem>

int main() {
    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;

    //std::string path = "images/watermark.png";

    //cv::Mat wm = cv::imread(path, cv::IMREAD_GRAYSCALE);

    //if (wm.empty()) {
    //    std::cerr << "error with loading wm!" << std::endl;
    //    return -1;
    //}
    //
    //cv::Mat trans_wm = affineTransform(wm, 2, 1, -1, 3, 5, 4);
    //cv::Mat rest_wm = affineTransformInv(trans_wm, 2, 1, -1, 3, 5, 4);
    //
    //cv::imwrite("images/trans_wm.png", trans_wm);
    //cv::imwrite("images/rest_wm.png", rest_wm);

    std::vector<int> keys = { 2, 1, -1, 3, 5, 4 };
    writeVectorToFile(keys);
    
    cv::Mat wm = cv::imread("images/watermark.png", cv::IMREAD_GRAYSCALE);
    //wm = affineTransform(wm, 2, 1, -1, 3, 5, 4);
    cv::Mat img = importImage("images/lenna.png");
   

    cv::Mat new_img = embedWatermark(img, wm, 52.9);
    exportImage(new_img, "images/new_lenna.png");

    cv::Mat new_imp_img = importImage("images/new_lenna.png");
    cv::Mat extractedWatermark = extractWatermark(new_imp_img, 52.9);
    //extractedWatermark = affineTransformInv(wm, 2, 1, -1, 3, 5, 4);
    exportImage(extractedWatermark, "images/new_wm.png");

    /*cv::Mat enc_wm = process(wm);
    exportImage(enc_wm, "images/enc_wm.png");
    cv::Mat dec_wm = restore(enc_wm);
    exportImage(dec_wm, "images/dec_wm.png");*/

    std::cout << "Done" << std::endl;
    std::cin.get();
    return 0;
}
