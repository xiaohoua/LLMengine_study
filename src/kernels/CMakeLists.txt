add_library(embeddingFunctor STATIC input_embedding.cu)
# 允许CUDA源文件被单独编译成.ptx（Parallel Thread Execution）代码，
# 这样可以加快大型项目中的编译速度，因为CUDA内核不需要在每次编译时都被重新编译。
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
# 编译器将生成位置无关代码（PIC），如果静态库或目标文件需要在可执行文件或动态库中被重定位。
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
# 允许链接器解析设备代码中的符号，通常用于处理静态链接的CUDA内核。
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(paddingoffsetFunctor STATIC cal_paddingoffset.cu)
set_property(TARGET paddingoffsetFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET paddingoffsetFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET paddingoffsetFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(rmsnormFunctor STATIC rmsnorm_kernel.cu)
set_property(TARGET rmsnormFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rmsnormFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rmsnormFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)