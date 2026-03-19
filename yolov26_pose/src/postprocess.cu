#include "postprocess.h"

// ===== Decode kernel =====
// 输入已经是 (max_det, 14) 排列，bbox 已是 xyxy，无需转置和 NMS
__global__ void decode_end2end_pose_kernel(
    const float*      src,          // [max_det, 14]
    int               max_det,      // 300
    float             conf_thresh,
    const pose::PreParam pparam,
    float*            out_boxes,    // [max_det, 6]  x0,y0,x1,y1,score,label
    float*            out_kps,      // [max_det, 8]  4个关键点 x,y
    int*              out_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_det) return;

    const float* p = src + idx * 14;

    // [x1, y1, x2, y2, conf, cls, kpt1x, kpt1y, kpt2x, kpt2y, kpt3x, kpt3y, kpt4x, kpt4y]
    float x0   = p[0];
    float y0   = p[1];
    float x1   = p[2];
    float y1   = p[3];
    float conf = p[4];
    float cls  = p[5];

    if (conf < conf_thresh) return;

    // 坐标还原：减 padding 偏移，再乘缩放比
    x0 = (x0 - pparam.dw) * pparam.ratio;
    y0 = (y0 - pparam.dh) * pparam.ratio;
    x1 = (x1 - pparam.dw) * pparam.ratio;
    y1 = (y1 - pparam.dh) * pparam.ratio;

    // clip 到原图范围
    x0 = fminf(fmaxf(x0, 0.f), pparam.width);
    y0 = fminf(fmaxf(y0, 0.f), pparam.height);
    x1 = fminf(fmaxf(x1, 0.f), pparam.width);
    y1 = fminf(fmaxf(y1, 0.f), pparam.height);

    if (x0 >= x1 || y0 >= y1) return;

    int index = atomicAdd(out_count, 1);
    if (index >= max_det) return;

    // 写入 box
    int box_base = index * 6;
    out_boxes[box_base + 0] = x0;
    out_boxes[box_base + 1] = y0;
    out_boxes[box_base + 2] = x1;
    out_boxes[box_base + 3] = y1;
    out_boxes[box_base + 4] = conf;
    out_boxes[box_base + 5] = cls;

    // 写入关键点：4个关键点，每个只有 x,y（无 visibility）
    int kps_base = index * 8;
    for (int k = 0; k < 4; ++k) {
        float kx = (p[6 + 2*k]     - pparam.dw) * pparam.ratio;
        float ky = (p[6 + 2*k + 1] - pparam.dh) * pparam.ratio;

        kx = fminf(fmaxf(kx, 0.f), pparam.width);
        ky = fminf(fmaxf(ky, 0.f), pparam.height);

        out_kps[kps_base + 2*k]     = kx;
        out_kps[kps_base + 2*k + 1] = ky;
    }
}

// ===== 主函数 =====
void cuda_postprocess(
    std::vector<pose::Object>& objs,
    const float*               d_output,
    int                        max_det,
    const pose::PreParam&      pparam,
    float                      score_thres)
{
    objs.clear();

    const int BLOCK = 256;
    int grid = (max_det + BLOCK - 1) / BLOCK;

    // 1. 分配输出缓冲
    float* d_boxes = nullptr;
    float* d_kps   = nullptr;
    int*   d_count = nullptr;
    CHECK(cudaMalloc(&d_boxes, max_det * 6 * sizeof(float)));
    CHECK(cudaMalloc(&d_kps,   max_det * 8 * sizeof(float)));
    CHECK(cudaMalloc(&d_count, sizeof(int)));
    CHECK(cudaMemset(d_boxes, 0, max_det * 6 * sizeof(float)));
    CHECK(cudaMemset(d_kps,   0, max_det * 8 * sizeof(float)));
    CHECK(cudaMemset(d_count, 0, sizeof(int)));

    // 2. Decode（无需转置，无需 NMS，end2end 已处理）
    decode_end2end_pose_kernel<<<grid, BLOCK>>>(
        d_output, max_det, score_thres, pparam,
        d_boxes, d_kps, d_count);
    CHECK(cudaDeviceSynchronize());

    // 3. 取有效数量
    int h_count = 0;
    CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_count <= 0) {
        cudaFree(d_boxes);
        cudaFree(d_kps);
        cudaFree(d_count);
        return;
    }
    h_count = std::min(h_count, max_det);

    // 4. D2H
    std::vector<float> h_boxes(h_count * 6);
    std::vector<float> h_kps(h_count * 8);
    CHECK(cudaMemcpy(h_boxes.data(), d_boxes, h_count * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_kps.data(),   d_kps,   h_count * 8 * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. 构造 Object
    for (int i = 0; i < h_count; ++i) {
        pose::Object obj;

        float x0 = h_boxes[i*6 + 0];
        float y0 = h_boxes[i*6 + 1];
        float x1 = h_boxes[i*6 + 2];
        float y1 = h_boxes[i*6 + 3];

        // common.hpp 字段名是 boxes
        obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
        obj.prob  = h_boxes[i*6 + 4];
        obj.label = static_cast<int>(h_boxes[i*6 + 5]);

        // 4个关键点，存为 [x0,y0, x1,y1, x2,y2, x3,y3]
        obj.kps.clear();
        obj.kps.reserve(8);
        for (int k = 0; k < 8; ++k) {
            obj.kps.push_back(h_kps[i * 8 + k]);
        }

        objs.push_back(obj);
    }

    // 6. 按置信度降序
    std::sort(objs.begin(), objs.end(),
        [](const pose::Object& a, const pose::Object& b) {
            return a.prob > b.prob;
        });

    // 7. 释放
    cudaFree(d_boxes);
    cudaFree(d_kps);
    cudaFree(d_count);
}