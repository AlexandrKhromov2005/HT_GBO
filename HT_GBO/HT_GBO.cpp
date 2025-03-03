#include <opencv2/opencv.hpp>
#include "src/affine.h"
#include "src/POB.h"
#include "src/image_processing.h"

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

    cv::Mat img = importImage("images/lenna.png");
    std::cout << "rows = " << img.rows << " cols = " << img.cols << "\n";

    cv::Mat mat = (cv::Mat_<uchar>(4, 4) <<
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160);
    
    std::cout << "element = " << (int)mat.at<uchar>(15) << "\n";

    //img = splitIntoBlocks(img);

    std::cin.get();
    return 0;
}