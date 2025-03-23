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
#include "src/test_attacks.h"

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
        
        GBO optimizer(host, wm);
        optimizer.optimize();

        std::cout << "T = " << optimizer.optimal_t << "\n";

        cv::Mat result = embedWatermark(host, wm, optimizer.optimal_t);
        /*cv::Mat result = embedWatermark(host, wm, 30);*/

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

        std::string t_str;
        std::cout << "Enter the quantization step:\n";
        std::cin >> t_str;

        double t = std::stod(t_str);

        cv::Mat image = importImage("images/" + image_name);
        cv::Mat wm = importImage("images/" + wm_name);

        cv::Mat new_image = importImage("images/" + new_image_name);
        cv::Mat ext_wm = importImage("images/" + ext_wm_name);


        double psnr =  computeImagePSNR(image, new_image);
        double ssim = computeImageSSIM(image, new_image);

        double ber = computeImageBER(wm, ext_wm);
        double ncc = computeImageNCC(wm, ext_wm);

        cv::Mat img_jp40 = jpegCompression40(new_image);
        ext_wm = extractWatermark(img_jp40, t);
        double ber_jp40 = computeImageBER(wm, ext_wm);
        double ncc_jp40 = computeImageNCC(wm, ext_wm);

        cv::Mat img_jp70 = jpegCompression70(new_image);
        ext_wm = extractWatermark(img_jp70, t);
        double ber_jp70 = computeImageBER(wm, ext_wm);
        double ncc_jp70 = computeImageNCC(wm, ext_wm);

        cv::Mat img_mfilt3x3 = medianFilter3x3(new_image);
        ext_wm = extractWatermark(img_mfilt3x3, t);
        double ber_mfilt3x3 = computeImageBER(wm, ext_wm);
        double ncc_mfilt3x3 = computeImageNCC(wm, ext_wm);

        cv::Mat img_gfilt5x5 = gaussianFilter5x5(new_image);
        ext_wm = extractWatermark(img_gfilt5x5, t);
        double ber_gfilt5x5 = computeImageBER(wm, ext_wm);
        double ncc_gfilt5x5 = computeImageNCC(wm, ext_wm);

        cv::Mat img_rot30 = rotate30(new_image);
        ext_wm = extractWatermark(img_rot30, t);
        double ber_rot30 = computeImageBER(wm, ext_wm);
        double ncc_rot30 = computeImageNCC(wm, ext_wm);

        cv::Mat img_eas4 = scale4(new_image);
        ext_wm = extractWatermark(img_eas4, t);
        double ber_eas4 = computeImageBER(wm, ext_wm);
        double ncc_eas4 = computeImageNCC(wm, ext_wm);

        cv::Mat img_tranclate20_40 = translate20_40(new_image);
        ext_wm = extractWatermark(img_tranclate20_40, t);
        double ber_tranclate20_40 = computeImageBER(wm, ext_wm);
        double ncc_tranclate20_40 = computeImageNCC(wm, ext_wm);

        std::cout << "PSNR for image = " << psnr << "\n";
        std::cout << "SSIM for image = " << ssim << "\n";

        std::cout << "WM BER before attack: " << ber << "\n";
        std::cout << "WM NCC before attack: " << ncc << "\n";

        std::cout << "WM mterics after attacks:\n";

        std::cout << "BER after JPEG 70: " << ber_jp70 << "\n";
        std::cout << "NCC after JPEG 70: " << ncc_jp70 << "\n";

        std::cout << "BER after Median filtering 3x3: " << ber_mfilt3x3 << "\n";
        std::cout << "NCC after Median filtering 3x3: " << ncc_mfilt3x3 << "\n";

        std::cout << "BER after Gaussian filtering 5x5: " << ber_gfilt5x5 << "\n";
        std::cout << "NCC after Gaussian filtering 5x5: " << ncc_gfilt5x5 << "\n";

        std::cout << "BER after Rotate correction 30: " << ber_rot30 << "\n";
        std::cout << "NCC after Rotate correction 30: " << ncc_rot30 << "\n";

        std::cout << "BER after Enlarge and correction 4: " << ber_eas4 << "\n";
        std::cout << "NCC after Enlarge and correction 4: " << ncc_eas4 << "\n";

        std::cout << "BER after Translation correction (20,40): " << ber_tranclate20_40 << "\n";
        std::cout << "NCC after Translation correction (20,40): " << ncc_tranclate20_40 << "\n";
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
