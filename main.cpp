#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>

#include "yolov26_pose/include/yolov26_pose.h"
#include "plate_recognition/include/plate_recognition.h"

struct PlateDet {
    cv::Rect_<float>         bbox;           // 检测框
    float                    confidence;     // 检测置信度
    int                      plate_type;     // 0=单层 1=双层
    std::vector<float>       key_points;     // 4个关键点 [x0,y0,x1,y1,x2,y2,x3,y3]
    std::string              plate_license;  // 车牌号
    std::string              plate_color;    // 颜色
    float                    color_conf;     // 颜色置信度
};

// landmarks 顺序：左上→右上→右下→左下
cv::Mat four_point_transform(const cv::Mat& src, const std::vector<float>& kps)
{
    // kps = [x0,y0, x1,y1, x2,y2, x3,y3]
    cv::Point2f tl(kps[0], kps[1]);
    cv::Point2f tr(kps[2], kps[3]);
    cv::Point2f br(kps[4], kps[5]);
    cv::Point2f bl(kps[6], kps[7]);

    float width_a  = cv::norm(br - bl);
    float width_b  = cv::norm(tr - tl);
    int   max_width = std::max((int)width_a, (int)width_b);

    float height_a  = cv::norm(tr - br);
    float height_b  = cv::norm(tl - bl);
    int   max_height = std::max((int)height_a, (int)height_b);

    if (max_width <= 0 || max_height <= 0) return cv::Mat();

    std::vector<cv::Point2f> src_pts = {tl, tr, br, bl};
    std::vector<cv::Point2f> dst_pts = {
        {0,                          0           },
        {(float)(max_width - 1),     0           },
        {(float)(max_width - 1),     (float)(max_height - 1)},
        {0,                          (float)(max_height - 1)},
    };

    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(src, warped, M, cv::Size(max_width, max_height));
    return warped;
}

// 双层车牌上下拼合
cv::Mat get_split_merge(const cv::Mat& img)
{
    int upper_h = (int)(5.0 / 12 * img.rows);
    int lower_y = (int)(1.0 / 3 * img.rows);
    int lower_h = img.rows - lower_y;

    cv::Rect upper_rect(0, 0,      img.cols, upper_h);
    cv::Rect lower_rect(0, lower_y, img.cols, lower_h);

    cv::Mat img_upper = img(upper_rect).clone();
    cv::Mat img_lower = img(lower_rect).clone();
    cv::resize(img_upper, img_upper, img_lower.size());

    cv::Mat out(img_lower.rows,
                img_lower.cols + img_upper.cols,
                CV_8UC3,
                cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,              0, img_upper.cols, img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols, 0, img_lower.cols, img_lower.rows)));
    return out;
}

// ===== 读取目录下的图片列表 =====
int readFileList(const std::string& path,
                 std::vector<std::string>& fileList,
                 const std::vector<std::string>& extensions)
{
    fileList.clear();
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory: " << path << std::endl;
        return -1;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            std::string filename = entry->d_name;
            size_t dot = filename.find_last_of('.');
            if (dot != std::string::npos) {
                std::string ext = filename.substr(dot + 1);
                for (const auto& e : extensions) {
                    if (ext == e) {
                        fileList.push_back(path + "/" + filename);
                        break;
                    }
                }
            }
        }
    }
    closedir(dir);
    std::sort(fileList.begin(), fileList.end());
    return 0;
}

// ===== 绘制结果 =====
void drawBboxes(cv::Mat& img,
                const std::vector<PlateDet>& dets,
                cv::Ptr<cv::freetype::FreeType2>& ft2)
{
    static cv::Scalar kp_colors[4] = {
        cv::Scalar(0,   255, 255),
        cv::Scalar(0,   0,   255),
        cv::Scalar(0,   255, 0  ),
        cv::Scalar(255, 0,   255),
    };

    for (int f = 0; f < (int)dets.size(); ++f) {
        const auto& d = dets[f];

        int x1 = (int)d.bbox.x;
        int y1 = (int)d.bbox.y;
        int x2 = (int)(d.bbox.x + d.bbox.width);
        int y2 = (int)(d.bbox.y + d.bbox.height);

        // 检测框
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(255, 0, 0), 2);

        // 4个关键点
        for (int k = 0; k < 4 && k * 2 + 1 < (int)d.key_points.size(); ++k) {
            int kx = (int)std::round(d.key_points[2 * k]);
            int ky = (int)std::round(d.key_points[2 * k + 1]);
            cv::circle(img, cv::Point(kx, ky), 4, kp_colors[k], -1);
        }

        // 标签文字
        std::string label = d.plate_license + " " + d.plate_color;
        if (d.plate_type == 1) label += " 双层";

        const int font_size = 18;
        const int pad_x     = 4;
        const int pad_y     = 3;

        // 估算文字宽度：
        // 中文字符（Unicode > 127）宽度 ≈ font_size
        // ASCII 字符宽度 ≈ font_size * 0.55
        int text_w = 0;
        for (size_t ci = 0; ci < label.size(); ) {
            unsigned char c = (unsigned char)label[ci];
            if (c >= 0x80) {
                // UTF-8 多字节字符（中文占3字节）
                text_w += font_size;
                ci += 3;
            } else {
                // ASCII
                text_w += (int)(font_size * 0.55f);
                ci += 1;
            }
        }
        int text_h = font_size;  // 行高直接用 font_size

        int box_w = text_w + pad_x * 2;
        int box_h = text_h + pad_y * 2;

        // 背景框位置：检测框上方
        int bg_x1 = x1;
        int bg_y1 = y1 - box_h - 2;
        if (bg_y1 < 0) bg_y1 = y1 + 2;
        int bg_x2 = bg_x1 + box_w;
        int bg_y2 = bg_y1 + box_h;

        // 背景框
        cv::rectangle(img,
            cv::Point(bg_x1, bg_y1),
            cv::Point(bg_x2, bg_y2),
            cv::Scalar(255, 255, 255), cv::FILLED);

        // 文字坐标：ft2 的 y 是基线，从背景框顶部 + pad_y + text_h
        int text_x = bg_x1 + pad_x;
        int text_y = bg_y1 + pad_y + text_h;

        if (ft2) {
            ft2->putText(img, label,
                cv::Point(text_x, text_y),
                font_size, cv::Scalar(0, 0, 0), -1, 8, true);
        } else {
            cv::putText(img, label,
                cv::Point(text_x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 1);
        }
    }
}

// ===== 主函数 =====
int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "  Plate Detection & Recognition System  " << std::endl;
    std::cout << "========================================" << std::endl;

    // ── 路径配置 ──
    std::string detect_engine = "../yolov26_pose/weights/yolo26s-plate-detect.engine";
    std::string rec_engine    = "../plate_recognition/weights/plate_rec_color.engine";
    std::string image_dir     = "../data";
    std::string output_dir    = "../result";
    std::string font_path     = "../font/NotoSansCJK-Regular.otf";

    if (argc > 1) detect_engine = argv[1];
    if (argc > 2) rec_engine    = argv[2];
    if (argc > 3) image_dir     = argv[3];
    if (argc > 4) output_dir    = argv[4];
    if (argc > 5) font_path     = argv[5];

    mkdir(output_dir.c_str(), 0755);

    // ── 初始化检测模型 ──
    std::cout << "\nLoading detector: " << detect_engine << std::endl;
    auto* detector = new YOLOv26_pose(detect_engine);
    detector->make_pipe(true);

    // ── 初始化识别模型 ──
    std::cout << "Loading recognizer: " << rec_engine << std::endl;
    bool is_color = true;
    auto* recognizer = new PlateRecognition(rec_engine);
    recognizer->make_pipe(true);

    // ── 加载字体 ──
    cv::Ptr<cv::freetype::FreeType2> ft2;
    try {
        ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData(font_path, 0);
        std::cout << "Font loaded: " << font_path << std::endl;
    } catch (const cv::Exception&) {
        std::cerr << "Warning: font not loaded, Chinese may not display" << std::endl;
        ft2 = nullptr;
    }

    // ── 读取图片列表 ──
    std::vector<std::string> imageList;
    std::vector<std::string> fileTypes{"jpg","png","jpeg","JPG","PNG","JPEG"};
    if (readFileList(image_dir, imageList, fileTypes) != 0 || imageList.empty()) {
        std::cerr << "Error: No images found in " << image_dir << std::endl;
        return -1;
    }
    std::cout << "Found " << imageList.size() << " images" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // ── 推理参数 ──
    float score_thres = 0.25f;
    int   total_plates    = 0;
    double total_det_ms   = 0.0;
    double total_rec_ms   = 0.0;
    int    processed      = 0;

    for (size_t i = 0; i < imageList.size(); ++i) {
        cv::Mat img = cv::imread(imageList[i]);
        if (img.empty()) {
            std::cerr << "[WARN] Cannot read: " << imageList[i] << std::endl;
            continue;
        }

        // ══════════════════════════════════
        // Step 1: 检测（YOLOv26-pose）
        // ══════════════════════════════════
        auto t0 = std::chrono::high_resolution_clock::now();

        detector->preprocessGPU(img);
        detector->infer();
        std::vector<pose::Object> objs;
        detector->postprocessGPU(objs, score_thres);

        auto t1 = std::chrono::high_resolution_clock::now();
        double det_ms = std::chrono::duration_cast<
            std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        total_det_ms += det_ms;

        // ══════════════════════════════════
        // Step 2: 识别（PlateRecognition）
        // ══════════════════════════════════
        std::vector<PlateDet> plate_dets;
        double rec_ms_total = 0.0;

        for (auto& obj : objs) {
            // 2.1 透视变换：用4个关键点矫正车牌区域
            //     obj.kps = [x0,y0, x1,y1, x2,y2, x3,y3]，对应左上→右上→右下→左下
            cv::Mat roi;
            if (obj.kps.size() >= 8) {
                roi = four_point_transform(img, obj.kps);
            } else {
                // 无关键点时退化为矩形裁剪
                cv::Rect rect = cv::Rect(
                    (int)obj.rect.x, (int)obj.rect.y,
                    (int)obj.rect.width, (int)obj.rect.height);
                rect &= cv::Rect(0, 0, img.cols, img.rows);
                if (rect.area() <= 0) continue;
                roi = img(rect).clone();
            }

            if (roi.empty()) continue;

            // 2.2 判断是否为双层车牌（plate_type 由检测 label 决定）
            // 对应 Python: if plate_type == 1: roi = get_split_merge(roi)
            int plate_type = obj.label;
            if (plate_type == 1) {
                roi = get_split_merge(roi);
            }

            // 2.3 识别
            auto r0 = std::chrono::high_resolution_clock::now();

            recognizer->copy_from_Mat(roi);
            recognizer->infer();
            plate_recogn::PlateResult rec_result = recognizer->postprocess(is_color);

            auto r1 = std::chrono::high_resolution_clock::now();
            rec_ms_total += std::chrono::duration_cast<
                std::chrono::microseconds>(r1 - r0).count() / 1000.0;

            // 2.4 组装结果
            PlateDet det;
            det.bbox          = obj.rect;
            det.confidence    = obj.prob;
            det.plate_type    = plate_type;
            det.key_points    = obj.kps;
            det.plate_license = rec_result.plate;
            det.plate_color   = rec_result.color;
            det.color_conf    = rec_result.color_conf;
            plate_dets.push_back(det);
        }

        total_rec_ms  += rec_ms_total;
        total_plates  += (int)plate_dets.size();
        processed++;

        // ── 控制台输出 ──
        std::cout << "[" << processed << "/" << imageList.size() << "] "
                  << imageList[i] << std::endl;
        std::cout << "  det=" << det_ms << "ms"
                  << " rec=" << rec_ms_total << "ms"
                  << " | plates=" << plate_dets.size() << std::endl;
        for (int j = 0; j < (int)plate_dets.size(); ++j) {
            std::cout << "  [" << j+1 << "] "
                      << plate_dets[j].plate_license
                      << " " << plate_dets[j].plate_color
                      << " (det:" << std::fixed << std::setprecision(2)
                      << plate_dets[j].confidence
                      << " col:" << plate_dets[j].color_conf << ")"
                      << (plate_dets[j].plate_type == 1 ? " 双层" : "")
                      << std::endl;
        }

        // ── 绘制并保存 ──
        drawBboxes(img, plate_dets, ft2);

        size_t last = imageList[i].find_last_of('/');
        std::string name = (last == std::string::npos)
                         ? imageList[i] : imageList[i].substr(last + 1);
        cv::imwrite(output_dir + "/" + name, img);
    }

    // ── 统计汇总 ──
    std::cout << "\n========================================" << std::endl;
    std::cout << "           Processing Summary           " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total images : " << processed    << std::endl;
    std::cout << "Total plates : " << total_plates << std::endl;
    if (processed > 0) {
        std::cout << "Avg det time : " << std::fixed << std::setprecision(1)
                  << total_det_ms / processed << " ms" << std::endl;
        std::cout << "Avg rec time : " << std::fixed << std::setprecision(1)
                  << total_rec_ms / processed << " ms" << std::endl;
        std::cout << "Avg total    : " << std::fixed << std::setprecision(1)
                  << (total_det_ms + total_rec_ms) / processed << " ms" << std::endl;
        std::cout << "FPS          : " << std::fixed << std::setprecision(2)
                  << 1000.0 * processed / (total_det_ms + total_rec_ms) << std::endl;
    }
    std::cout << "Results saved: " << output_dir << std::endl;
    std::cout << "========================================" << std::endl;

    delete detector;
    delete recognizer;
    return 0;
}