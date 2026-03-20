#ifndef POSE_NORMAL_YOLOv26_pose_HPP
#define POSE_NORMAL_YOLOv26_pose_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include <fstream>

const std::vector<std::vector<unsigned int>> KPS_COLORS = {
    {255, 0,   0  },   // kpt0: 蓝
    {0,   255, 0  },   // kpt1: 绿
    {0,   0,   255},   // kpt2: 红
    {0,   255, 255},   // kpt3: 黄
};

class YOLOv26_pose {
public:
    explicit YOLOv26_pose(const std::string& engine_file_path);
    ~YOLOv26_pose();

    void make_pipe(bool warmup = true);

    // CPU 预处理（letterbox + blobFromImage）
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    // GPU 预处理（CUDA kernel）
    void preprocessGPU(const cv::Mat& image);

    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void infer();

    void postprocess(
        std::vector<pose::Object>& objs,
        float score_thres = 0.25f,
        int   topk        = 100);

    void postprocessGPU(
        std::vector<pose::Object>& objs,
        float score_thres = 0.25f);
    
    static void draw_objects(
        const cv::Mat&                                image,
        cv::Mat&                                      res,
        const std::vector<pose::Object>&              objs,
        const std::vector<std::vector<unsigned int>>& KPS_COLORS
    );

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    int                  dst_w = 640;
    int                  dst_h = 640;
    std::vector<pose::Binding> input_bindings;
    std::vector<pose::Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    pose::PreParam       pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif  // POSE_NORMAL_YOLOv26_pose_HPP
