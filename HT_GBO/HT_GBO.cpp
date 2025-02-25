﻿#include <opencv2/opencv.hpp>
#include "src/affine.h"
#include "src/POB.h"


int main() {

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

    for (unsigned short i = 0; i < 16; ++i) {
        std::pair<int, int> res = pob(i);
        std::cout << i << " " << res.first << " " << res.second << std::endl;
   
    }

    return 0;
}