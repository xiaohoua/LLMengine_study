cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(oneLLM LANGUAGE CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
# 把路径${CUDA_PATH}/lib64添加到CMake的模块搜索路径中
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
# find_package(CUDA REQUIRED)

# setting compiler flags
# CMAKE_C_FLAGS是CMake根据系统环境和所选择的生成器（generator）来决定默认的编译器标志
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
# -Wall 开启所有警告
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
#cmake构建过程中输出信息
message("-- Assign GPU architecture (sm=70 75 80 86)")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
# -O0禁用优化 -G启用cuda调试信息
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")
#STATUS关键字，输出的信息会被标记为状态信息
message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_STANDARD 11)
# 如果编译器不支持指定的c++版本就报错
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "11")
    # 告诉CUDA编译器（nvcc）启用实验性的扩展lambda功能
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    # 允许在CUDA内核中使用更宽松的constexpr规则
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    # 虽然nvcc有自己的语法和规则，但--std=c++11标志告诉它在处理非CUDA相关的C++代码时遵循C++11标准。
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()

# O3最高级别的优化
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")
# 控制构建产物（如静态库、动态库和可执行文件）的输出目录
# 如果指定了构建目录（一般是build）CMAKE_BINARY_DIR就是build
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)
# 允许源代码根据编译时的配置做出不同的行为。 
# 例如，如果PERF选项被设置为ON，那么在编译时，所有源文件都将包含#define PERF
option (PERF
  "measure model inference performance"
  OFF
)
option (PRINT_DATA
  "print kernel output to debug"
  OFF
)
option (SAVE_DATA
  "save kernel output to debug"
  OFF
)
if (PERF)
    add_compile_options(-DPERF)
endif()
if (PRINT_DATA)
    add_compile_options(-DPRINT_DATA)
endif()
if (SAVE_DATA)
    add_compile_options(-DSAVE_DATA)
endif()

add_subdirectory(src)
add_subdirectory(tests)