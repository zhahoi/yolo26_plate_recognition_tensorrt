#include "plate_recognition.h"

PlateRecognition::PlateRecognition(const std::string& engine_file_path)
{
    // 1. 打开 TensorRT 引擎文件（.engine / .trt）, 以二进制模式读取
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good()); // 确保文件打开成功

    // 2. 定位到文件末尾，获取文件大小
    file.seekg(0, std::ios::end);
    auto size = file.tellg(); // size = 文件字节数
    file.seekg(0, std::ios::beg); // 回到文件开头

    // 3. 在堆上分配一个char数组用于存放文件内容，并读取
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close(); // 关闭文件

    // 4. 初始化TensorRT的自定义Plugin(若使用了任何自定义Layer)
    initLibNvInferPlugins(&this->gLogger, "");

    // 5. 创建运行时对象（IRuntime），后续用于反序列化 Engine
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 6. 反序列化 Engine，将二进制流构建为 ICudaEngine 实例
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);

    // 7. 释放已经不再需要的模型流内存
    delete[] trtModelStream;

    // 8. 基于 Engine 创建执行上下文（IExecutionContext），管理推理状态
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 9. 创建一个 CUDA Stream，用于后续异步拷贝与推理
    cudaStreamCreate(&this->stream);

    // 10. 查询模型中所有的 I/O Tensor 数量（输入 + 输出）
    this->num_bindings = this->engine->getNbIOTensors();

    // 11. 遍历每个 binding，收集其名称、数据类型、尺寸，以及区分输入/输出
    for (int i = 0; i < this->num_bindings; ++i) {
        plate_recogn::Binding binding;

        // 11.1 获取第 i 个 binding 的名称（如 "images"、"output0"）
        std::string name = this->engine->getIOTensorName(i);
        binding.name = name;

        // 11.2 获取该 Tensor 的数据类型（float、int8…），并计算每个元素所占字节大小
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
        binding.dsize = type_to_size(dtype);

        // 11.3 判断该 binding 是否为输入
        bool IsInput = this->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

        // 11.4 获取该 Tensor 在最优配置下的最大 shape（动态 shape 时使用）
        nvinfer1::Dims     dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

        if (IsInput) {
            // ---- 处理输入 binding ----
            this->num_inputs += 1;

            // 11.5 计算总元素数量
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape

            // 11.6 告诉执行上下文：输入 Tensor 的 shape，供后续动态推理使用
            this->context->setInputShape(name.c_str(), dims);

            std::cout << "input name: " << name << " dims: " << dims.nbDims
                << " input shape:(" << dims.d[0] << "," << dims.d[1] << ","
                << dims.d[2] << "," << dims.d[3] << ")" << std::endl;
        }
        else {
            // ---- 处理输出 binding ----
            // 输出的 shape 可以通过上下文直接查询（动态 shape 时已被固定）
            dims = this->context->getTensorShape(name.c_str());
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;

            std::cout << "ouput name: " << name << " dims: " << dims.nbDims
                << " ouput shape:(" << dims.d[0] << "," << dims.d[1] << ","
                << dims.d[2] << "," << dims.d[3] << ")" << std::endl;
        }
    }
}

PlateRecognition::~PlateRecognition()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;

    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void PlateRecognition::make_pipe(bool warmup)
{

    // —— 1. 为所有输入 Binding 分配 Device (GPU) 内存 —— 
    // input_bindings 存储了每个输入张量的元素数量 (binding.size)
    // 和每个元素的字节大小 (binding.dsize)
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        // 异步地在当前 CUDA 流上分配 GPU 内存：bindings.size * bindings.dsize 字节 
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        // 将分配好的设备指针保存到 device_ptrs 以备后续推理使用
        this->device_ptrs.push_back(d_ptr);
    }

    // —— 2. 为所有输出 Binding 分配 Device 和 Host (Page‑locked) 内存 —— 
    // 输出通常要在 GPU 上计算后再拷贝回 CPU 端做后处理，
    // 所以既需要 Device 内存，也需要 Host 端的 page‑locked (pinned) 内存以加速复制
    for (auto& bindings : this->output_bindings) {
        void* d_ptr, * h_ptr;
        size_t size = bindings.size * bindings.dsize;
        // 在 GPU 上分配同样大小的输出 buffer
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        // 在 Host 上分配 page‑locked 内存，以便后续 cudaMemcpyAsync 高效地从 Device 读出
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // —— 3. 可选的模型“热身”——当 warmup 为 true 时运行若干次推理
    // 这样可以让 TensorRT JIT 编译、GPU 占用和内存分配提前完成，
    // 减少第一次真实推理的延迟峰值
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            // 3.1 对每个输入执行一次空数据拷贝，模拟真实输入
            for (auto& bindings : this->input_bindings) {
                size_t size = bindings.size * bindings.dsize;
                // 在 CPU 申请一个临时 buffer，并置零
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                // 异步把“全零”数据拷贝到 GPU 上的输入 buffer
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            // 3.2 调用 infer() 完成一次完整推理
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void PlateRecognition::copy_from_Mat(const cv::Mat& image)
{
    // 1. resize 到 (216, 48)
    cv::Mat resized;
    // 对小图用 INTER_CUBIC，比默认的 INTER_LINEAR 保留更多细节
    int interp = (image.cols < PLATE_INPUT_W || image.rows < PLATE_INPUT_H)
                 ? cv::INTER_CUBIC
                 : cv::INTER_LINEAR;
    cv::resize(image, resized, cv::Size(PLATE_INPUT_W, PLATE_INPUT_H), 0, 0, interp);

    // 2. 转 float，归一化：(pixel/255 - 0.588) / 0.193
    int channel_size = PLATE_INPUT_H * PLATE_INPUT_W;
    std::vector<float> data(3 * channel_size);

    for (int row = 0; row < PLATE_INPUT_H; ++row) {
        const uchar* uc_pixel = resized.data + row * resized.step;
        for (int col = 0; col < PLATE_INPUT_W; ++col) {
            int i = row * PLATE_INPUT_W + col;
            data[i]                    = ((float)uc_pixel[0] / 255.0f - PLATE_MEAN) / PLATE_STD; // B
            data[i + channel_size]     = ((float)uc_pixel[1] / 255.0f - PLATE_MEAN) / PLATE_STD; // G
            data[i + 2 * channel_size] = ((float)uc_pixel[2] / 255.0f - PLATE_MEAN) / PLATE_STD; // R
            uc_pixel += 3;
        }
    }

    // 3. H2D 拷贝
    size_t size = input_bindings[0].size * input_bindings[0].dsize;
    CHECK(cudaMemcpyAsync(
        device_ptrs[0], data.data(),
        size, cudaMemcpyHostToDevice, stream));
}

void PlateRecognition::infer()
{
    // 1. 绑定所有输入输出 Tensor 的 GPU 地址到执行上下文
    for (int i = 0; i < this->num_bindings; ++i) {
        const std::string& name = (i < this->num_inputs)
            ? this->input_bindings[i].name
            : this->output_bindings[i - this->num_inputs].name;
        this->context->setTensorAddress(name.c_str(), this->device_ptrs[i]);
    }

    // 2. 异步推理
    this->context->enqueueV3(this->stream);

    // 3. 将所有输出从 GPU 异步拷贝回 Host pinned 内存
    //    输出 binding 对应：
    //      host_ptrs[0] → ocr_output  [1, 27, 78]
    //      host_ptrs[1] → color_output [1, 5]
    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size
                     * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i],
            this->device_ptrs[i + this->num_inputs],
            osize,
            cudaMemcpyDeviceToHost,
            this->stream));
    }

    // 4. 等待当前 stream 上所有操作完成
    CHECK(cudaStreamSynchronize(this->stream));
}

plate_recogn::PlateResult PlateRecognition::postprocess(bool is_color)
{
    plate_recogn::PlateResult result;

    // ── Step1：按名称匹配输出节点 ──
    float* ocr_ptr   = nullptr;
    float* color_ptr = nullptr;
    for (int i = 0; i < (int)output_bindings.size(); ++i) {
        const std::string& name = output_bindings[i].name;
        if (name == "ocr_output") {
            ocr_ptr   = static_cast<float*>(host_ptrs[i]);
        } else if (name == "color_output") {
            color_ptr = static_cast<float*>(host_ptrs[i]);
        }
    }

    // ── Step2：名称匹配失败时按 size 匹配 ──
    if (!ocr_ptr || !color_ptr) {
        for (int i = 0; i < (int)output_bindings.size(); ++i) {
            size_t sz = output_bindings[i].size;
            if (sz == (size_t)(PLATE_SEQ_LEN * PLATE_NUM_CLASSES)) {
                ocr_ptr = static_cast<float*>(host_ptrs[i]);
            } else if (sz == (size_t)PLATE_COLOR_NUM) {
                color_ptr = static_cast<float*>(host_ptrs[i]);
            }
        }
    }

    if (!ocr_ptr) {
        std::cerr << "[ERROR] ocr_output not found" << std::endl;
        return result;
    }

    // ── Step3：OCR 解码 ──
    std::vector<int>   argmax_seq(PLATE_SEQ_LEN);
    std::vector<float> max_probs(PLATE_SEQ_LEN);

    for (int t = 0; t < PLATE_SEQ_LEN; ++t) {
        // 复制一行，softmax 不影响原始数据
        std::vector<float> row_vec(
            ocr_ptr + t * PLATE_NUM_CLASSES,
            ocr_ptr + t * PLATE_NUM_CLASSES + PLATE_NUM_CLASSES);

        // argmax 从原始 logits 取
        int   best_idx   = 0;
        float best_logit = row_vec[0];
        for (int c = 1; c < PLATE_NUM_CLASSES; ++c) {
            if (row_vec[c] > best_logit) {
                best_logit = row_vec[c];
                best_idx   = c;
            }
        }
        argmax_seq[t] = best_idx;

        // softmax 后取归一化概率用于显示
        softmax(row_vec.data(), PLATE_NUM_CLASSES);
        max_probs[t] = row_vec[best_idx];
    }

    // ── Step4：CTC 解码 ──
    std::vector<int> char_indices, char_positions;
    decodePlate(argmax_seq.data(), PLATE_SEQ_LEN, char_indices, char_positions);

    for (int i = 0; i < (int)char_indices.size(); ++i) {
        result.plate += PLATE_CHR[char_indices[i]];
        result.char_probs.push_back(max_probs[char_positions[i]]);
    }

    // ── Step5：颜色解码 ──
    if (is_color && color_ptr) {
        // argmax 从原始 logits 取
        int   best_color = 0;
        float best_logit = color_ptr[0];
        for (int c = 1; c < PLATE_COLOR_NUM; ++c) {
            if (color_ptr[c] > best_logit) {
                best_logit = color_ptr[c];
                best_color = c;
            }
        }

        // softmax 后取归一化概率
        std::vector<float> color_vec(
            color_ptr, color_ptr + PLATE_COLOR_NUM);
        softmax(color_vec.data(), PLATE_COLOR_NUM);

        result.color      = PLATE_COLOR[best_color];
        result.color_conf = color_vec[best_color];
    }

    return result;
}

void PlateRecognition::decodePlate(
    const int* preds, int seq_len,
    std::vector<int>& out_indices,
    std::vector<int>& out_positions)
{
    int pre = 0;
    for (int i = 0; i < seq_len; ++i) {
        if (preds[i] != 0 && preds[i] != pre) {
            out_indices.push_back(preds[i]);
            out_positions.push_back(i);
        }
        pre = preds[i];
    }
}

void PlateRecognition::softmax(float* data, int len)
{
    float max_val = *std::max_element(data, data + len);
    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < len; ++i) data[i] /= sum;
}