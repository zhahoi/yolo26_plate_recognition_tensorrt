cmake_minimum_required(VERSION 3.18)

# -------------------- CUDA 编译器 --------------------
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)

project(plate_recognition LANGUAGES CXX CUDA)

# -------------------- 编译标准 --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_definitions(-pthread)

# -------------------- OpenCV --------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------------- CUDA --------------------
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# -------------------- TensorRT --------------------
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

# -------------------- 项目头文件路径 --------------------
# 顶层 main.cpp 需要能找到两个子模块的 include
include_directories(
    ${PROJECT_SOURCE_DIR}/include 
    ${PROJECT_SOURCE_DIR}/yolov26_pose/include
    ${PROJECT_SOURCE_DIR}/plate_recognition/include
)

# -------------------- 源文件收集 --------------------
# 顶层 main.cpp
set(MAIN_SRC
    ${PROJECT_SOURCE_DIR}/main.cpp
)

# yolov26_pose 子模块：cpp + cu
set(YOLO_CPP_FILES
    ${PROJECT_SOURCE_DIR}/yolov26_pose/src/yolov26_pose.cpp
)
set(YOLO_CUDA_FILES
    ${PROJECT_SOURCE_DIR}/yolov26_pose/src/preprocess.cu
    ${PROJECT_SOURCE_DIR}/yolov26_pose/src/postprocess.cu
)

# plate_recognition 子模块：仅 cpp（无 cu 文件）
set(PLATE_CPP_FILES
    ${PROJECT_SOURCE_DIR}/plate_recognition/src/plate_recognition.cpp
)

# -------------------- 可执行文件 --------------------
add_executable(plate_recognition
    ${MAIN_SRC}
    ${YOLO_CPP_FILES}
    ${YOLO_CUDA_FILES}
    ${PLATE_CPP_FILES}
)

# -------------------- CUDA 文件属性 --------------------
set_source_files_properties(
    ${YOLO_CUDA_FILES}
    PROPERTIES
    LANGUAGE CUDA
    CUDA_PTX_COMPILATION       OFF
    CUDA_SEPARABLE_COMPILATION ON
)

# -------------------- Jetson NX 架构设置 --------------------
set_target_properties(plate_recognition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         "72"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# -------------------- 编译选项 --------------------
target_compile_options(plate_recognition PRIVATE
    # CUDA 性能优化（Jetson NX compute 7.2）
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=compute_72>
    $<$<COMPILE_LANGUAGE:CUDA>:--gpu-code=sm_72>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>
)

# -------------------- 编译宏定义 --------------------
target_compile_definitions(plate_recognition PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
)

# -------------------- 链接库 --------------------
target_link_libraries(plate_recognition
    PRIVATE
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
    # freetype 支持中文字体渲染
    opencv_freetype
    pthread
)

# -------------------- 可选：CUDA Profiling --------------------
option(ENABLE_CUDA_PROFILING "Enable CUDA profiling" OFF)
if(ENABLE_CUDA_PROFILING)
    target_compile_definitions(plate_recognition PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_PROFILING>
    )
endif()

# -------------------- 打印配置信息 --------------------
message(STATUS "========================================")
message(STATUS "  plate_recognition build configuration")
message(STATUS "========================================")
message(STATUS "  OpenCV version : ${OpenCV_VERSION}")
message(STATUS "  CUDA version   : ${CUDA_VERSION}")
message(STATUS "  CUDA arch      : 72 (Jetson NX)")
message(STATUS "  Source files   :")
message(STATUS "    main         : ${MAIN_SRC}")
message(STATUS "    yolo cpp     : ${YOLO_CPP_FILES}")
message(STATUS "    yolo cuda    : ${YOLO_CUDA_FILES}")
message(STATUS "    plate cpp    : ${PLATE_CPP_FILES}")
message(STATUS "========================================")