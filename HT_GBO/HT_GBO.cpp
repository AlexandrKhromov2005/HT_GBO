#include <opencv2/opencv.hpp>
#include "src/affine.h"
#include "src/POB.h"
#include "src/image_processing.h"
#include "src/processWM.h"
#include <filesystem>
#include "src/POB.h"
#include "src/Hadamard.h"
#include "src/GBO.h"

int main() {
    cv::Mat host = importImage("images/lenna_rgb.png");
    cv::Mat wm = importImage("images/pinguin.png");
    GBO optimizer(host, wm);
    optimizer.optimize();
    cv::Mat result = embedWatermark(host, wm, optimizer.optimal_t);

    std::cout << "T = " << optimizer.optimal_t << "\n";

    std::cout << "Done" << std::endl;
    std::cin.get();
    return 0;
}
