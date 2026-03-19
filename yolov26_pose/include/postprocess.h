#ifndef POSE_NORMAL_POSTPROCESS_CUH
#define POSE_NORMAL_POSTPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "common.hpp"


void cuda_postprocess(
    std::vector<pose::Object>& objs, 
    const float* d_output, 
    int max_det,
    const pose::PreParam& pparam, 
    float score_thres);

#endif // POSE_NORMAL_POSTPROCESS_CUH