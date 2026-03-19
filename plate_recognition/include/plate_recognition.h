#ifndef PLATE_RECOGNITION_H
#define PLATE_RECOGNITION_H

#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

static const std::vector<std::string> PLATE_COLOR = {
    "黑色", "蓝色", "绿色", "白色", "黄色"
};

static const std::vector<std::string> PLATE_CHR = {
    "#","京","沪","津","渝","冀","晋","蒙","辽","吉",
    "黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘",
    "粤","桂","琼","川","贵","云","藏","陕","甘","青",
    "宁","新","学","警","港","澳","挂","使","领","民",
    "航","危","0","1","2","3","4","5","6","7",
    "8","9","A","B","C","D","E","F","G","H",
    "J","K","L","M","N","P","Q","R","S","T",
    "U","V","W","X","Y","Z","险","品"
}; 

class PlateRecognition {

public:
    explicit PlateRecognition(const std::string& engine_file_path);
    ~PlateRecognition();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void infer();

    plate_recogn::PlateResult postprocess(bool is_color = true);

    static void decodePlate(
        const int*          preds,        // argmax 后的索引序列 [SEQ_LEN]
        int                 seq_len,
        std::vector<int>&   out_indices,  // 有效字符索引
        std::vector<int>&   out_positions // 对应原始位置（用于取概率）
    );

    static void softmax(float* data, int len);

public:
    int                       num_bindings;
    int                       num_inputs  = 0;
    int                       num_outputs = 0;
    std::vector<plate_recogn::Binding> input_bindings;
    std::vector<plate_recogn::Binding> output_bindings;
    std::vector<void*>        host_ptrs;
    std::vector<void*>        device_ptrs;

private:
    static constexpr float PLATE_MEAN = 0.588f;
    static constexpr float PLATE_STD  = 0.193f;

    static constexpr int PLATE_INPUT_W = 168;
    static constexpr int PLATE_INPUT_H = 48;

    static constexpr int PLATE_NUM_CLASSES = 78;   // 字符类别数
    static constexpr int PLATE_SEQ_LEN     = 21;   // 时间步长度
    static constexpr int PLATE_COLOR_NUM   = 5;    // 颜色类别数

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif // PLATE_RECOGNITION_H