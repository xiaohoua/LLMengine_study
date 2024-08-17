#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator
{
public:
    // BaseAllocator() = default; 默认的可以不写
    virtual ~BaseAllocator(){};//八股 一般要把父类 的构造函数设置为虚函数
    template<typename T>
    T* Malloc(T* ptr, size_t size, bool is_host){
        return (T*)UnifyMalloc((void*)ptr, size, is_host);
        //为什么用void* ：malloc返回的是void*
    }
    virtual void* UnifyMalloc(void* ptr, size_t size, bool is_host=false) = 0;//纯虚函数：子类必须实现这个函数
    template<typename T>
    void Free(T* ptr, bool is_host = false){
        UnifyFree((void*)ptr, is_host);
    }
    virtual void UnifyFree(void* ptr, bool is_host = false) = 0;
}