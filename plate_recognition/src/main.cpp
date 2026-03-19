#include "opencv2/opencv.hpp"
#include "plate_recognition.h"
#include <chrono>
#include <iomanip>

int main(int argc, char** argv)
{
    // 需要两个参数：权重文件路径、图片路径
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <tensorrt_engine.trt> <image_path>" << std::endl;
        return -1;
    }

    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string image_path{argv[2]};

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded: " << image_path
              << " (" << image.cols << "x" << image.rows << ")" << std::endl;

    bool is_color = true;
    auto plate_recogn = new PlateRecognition(engine_file_path);
    plate_recogn->make_pipe(true);

    // infer
    auto t0 = std::chrono::high_resolution_clock::now();

    plate_recogn->copy_from_Mat(image);
    plate_recogn->infer();
    plate_recogn::PlateResult result = plate_recogn->postprocess(is_color);

    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count() / 1000.0;

    std::cout << "=============================" << std::endl;
    std::cout << "Plate  : " << result.plate     << std::endl;
    if (is_color) {
        std::cout << "Color  : " << result.color
                  << " (" << std::fixed << std::setprecision(3)
                  << result.color_conf << ")"    << std::endl;
    }
    std::cout << "Probs  : ";
    for (float p : result.char_probs) {
        std::cout << std::fixed << std::setprecision(3) << p << " ";
    }
    std::cout << std::endl;
    std::cout << "Time   : " << infer_ms << " ms" << std::endl;
    std::cout << "=============================" << std::endl;

    // ── 释放资源 ──
    delete plate_recogn;
    return 0;
}