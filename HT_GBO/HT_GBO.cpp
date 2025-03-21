#include <opencv2/opencv.hpp>
#include "src/affine.h"
#include "src/POB.h"
#include "src/image_processing.h"
#include "src/processWM.h"
#include <filesystem>
#include "src/POB.h"
#include "src/Hadamard.h"
#include "src/GBO.h"
#include "src/image_metrics.h"

int main() {
    std::string choice;
    std::cout << "Select function:\n" << "0) Embed watermark\n" << "1) Extract watermark\n" << "2) Calculate metrics\n";
    std::cin >> choice;

    if (choice == "0") {
        std::string image_name;
        std::cout << "Enter the name of the image:\n";
        std::cin >> image_name;

        std::string wm_name;
        std::cout << "Enter the name of the watermark:\n";
        std::cin >> wm_name;

        std::string new_image_name;
        std::cout << "Enter the name of the watermarked image:\n";
        std::cin >> new_image_name;

        std::vector<int> keys = { 2, 1, -1, 3, 5, 4 };
        writeVectorToFile(keys);


        cv::Mat host = importImage("images/" + image_name);
        cv::Mat wm = importImage("images/" + wm_name);
        
        //GBO optimizer(host, wm);
        //optimizer.optimize();
        cv::Mat result = embedWatermark(host, wm, 46);

        exportImage(result, "images/" + new_image_name);
    }

    if (choice == "1") {
        std::string image_name;
        std::cout << "Enter the name of the watermarked image:\n";
        std::cin >> image_name;

        std::string wm_name;
        std::cout << "Enter the name of the extracted wateermark:\n";
        std::cin >> wm_name;

        std::string t_str;
        std::cout << "Enter the quantization step:\n";
        std::cin >> t_str;

        double t = std::stod(t_str);

        cv::Mat host = importImage("images/" + image_name);
        cv::Mat result = extractWatermark(host, t);
        exportImage(result, "images/" + wm_name);
    }

    if (choice == "2") {
        std::string image_name;
        std::cout << "Enter the name of the image:\n";
        std::cin >> image_name;

        std::string wm_name;
        std::cout << "Enter the name of the watermark:\n";
        std::cin >> wm_name;

        std::string new_image_name;
        std::cout << "Enter the name of the watermarked image:\n";
        std::cin >> new_image_name;

        std::string ext_wm_name;
        std::cout << "Enter the name of the extracted watermark:\n";
        std::cin >> ext_wm_name;

        cv::Mat image = importImage("images/" + image_name);
        cv::Mat wm = importImage("images/" + wm_name);

        cv::Mat new_image = importImage("images/" + new_image_name);
        cv::Mat ext_wm = importImage("images/" + ext_wm_name);

        double mse = computeImageMSE(image, new_image);
        double psnr =  computeImagePSNR(image, new_image);
        double ber = computeImageBER(wm, ext_wm);
        double ncc = computeImageNCC(wm, ext_wm);
        //double ssim = computeImageSSIM(const cv::Mat & img1, const cv::Mat & img2);

        std::cout << "MSE = " << mse << "\n";
        std::cout << "PSNR = " << psnr << "\n";
        std::cout << "BER = " << ber << "\n";
        std::cout << "NCC = " << ncc << "\n";
    }

    /*std::vector<int> keys = { 2, 1, -1, 3, 5, 4 };
    writeVectorToFile(keys);

    cv::Mat host = importImage("images/lenna_rgb.png");
    cv::Mat wm = importImage("images/pinguin.png");
    GBO optimizer(host, wm);
    optimizer.optimize();
    cv::Mat result = embedWatermark(host, wm, optimizer.optimal_t);

    std::cout << "T = " << optimizer.optimal_t << "\n";*/

    std::cout << "Done" << std::endl;
    std::cin.get();
    return 0;
}
